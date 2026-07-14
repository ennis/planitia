//! Document nodes.
//!
//! # Document
//!
//! A document is a tree of [Node](Node)s. Conceptually, nodes are composed of named _fields_,
//! which are either
//! - primitive values (e.g. string, number, boolean),
//! - child nodes (a single node, or a collection of nodes), or
//! - references to other nodes (a single node, or a collection of nodes).
//!
//! Nodes reference a [NodeDecl](NodeDecl) declaration, which specifies the expected fields, their names, and their kinds.
//!
//! # Nodes
//!
//! Nodes are allocated on the heap, with `Box::new`.
//! Internally, they are referenced via raw `*const Node` pointers for convenience, and their fields
//! have interior mutability via `Cell`.
//!
//! # Cursors
//!
//! To hold a reference to a node across edits, use `Cursor`. It is a safe handle to a node
//! that is invalidated when the node is removed from the document tree.

use std::alloc::{alloc, Layout};
use crate::decl::{FieldDecl, Multiplicity, NodeDecl, Type};
use aliasable::boxed::AliasableBox;
use slotmap::{new_key_type, SlotMap};
use std::cell::{Cell, RefCell, UnsafeCell};
use std::ops::Index;
use std::ptr;
use std::ptr::NonNull;
use crate::layout::LayoutItem;

pub type DocumentBox = AliasableBox<Document>;

/// Represents a document.
pub struct Document {
    /// Root node.
    root: Cell<*const Node>,
    /// Previous versions of the document tree, for undo/redo.
    history: Vec<HistoryEntry>,
    /// Index of the current version in the history.
    history_index: usize,
    pending_changes: RefCell<Vec<Edit>>,
    cursors: SlotMap<Cursor, CursorState>,
}

impl Document {
    fn invalidate_cursors(&mut self, node: NonNull<Node>) {
        self.cursors.retain(|_key, state| !ptr::addr_eq(state.node.as_ptr(), node.as_ptr()));
    }

    /// Creates a new document with the specified root node.
    pub fn new(root: &'static NodeDecl<'static>) -> DocumentBox {
        // Not `Box` because we don't want to invalidate pointers (`Node::document`).
        let this = AliasableBox::from_unique(Box::new(Document {
            root: Cell::new(ptr::null::<[NodeHeader; 0]>() as *const [NodeHeader] as *const Node),
            history: Vec::new(),
            history_index: 0,
            pending_changes: RefCell::new(Vec::new()),
            cursors: SlotMap::with_key(),
        }));

        {
            let root_node = Node::new(&*this, root);
            this.root.set(root_node);
        }
        this
    }

    /// Creates a cursor pointing to the specified node.
    pub fn create_cursor(&mut self, node: &Node) -> Cursor {
        let state = CursorState { node: NonNull::from_ref(node) };
        self.cursors.insert(state)
    }

    /// Removes a cursor.
    pub fn remove_cursor(&mut self, cursor: Cursor) {
        self.cursors.remove(cursor);
    }

    /// Creates a new node.
    pub fn create_node<'a>(&'a self, decl: &'static NodeDecl<'static>) -> &'a Node {
        unsafe {
            // SAFETY: The lifetime of the returned node is tied to the document.
            //         This prevents the document from being dropped while the reference is still alive,
            //         and prevents mut access to the document.
            &*Node::new(self, decl)
        }
    }

    /// Returns a reference to the root node of the document.
    pub fn root(&self) -> &Node {
        // SAFETY: the root pointer is valid for the lifetime of the document.
        unsafe { &*self.root.get() }
    }

    /// Returns a reference to the node that the cursor points to, if it is still valid.
    pub fn get(&self, cursor: Cursor) -> Option<&Node> {
        self.cursors.get(cursor).map(|state| unsafe { state.node.as_ref() })
    }

    /// Applies pending edits.
    // NOTE: in order for the assumptions made by unsafe code to be valid, the document should be mutated
    //       through this method only.
    pub fn apply_pending_changes(&mut self) {
        let changes = self.pending_changes.take();
        let history_entry = HistoryEntry { changes };
        for change in history_entry.changes.iter() {
            let parent = unsafe { &*change.node };

            match change.kind {
                EditKind::Insert { node, index_in_collection } => {
                    let collection = parent.children[change.field].as_collection().expect("invalid insert target");
                    unsafe {
                        // SAFETY: the reference to the node is valid for the lifetime of the collection.
                        (*collection.nodes.get()).insert(index_in_collection, node);

                        // SAFETY: No iterators to the collection are alive.
                        let node = node.as_ref();

                        // Set parent.
                        assert_eq!(node.h.parent.get(), None);
                        node.h.parent.set(Some((NonNull::from_ref(parent), change.field)));
                    }
                }
                EditKind::Remove { node, .. } => {
                    self.invalidate_cursors(node);
                    let collection = parent.children[change.field].as_collection().expect("invalid remove target");
                    let node_index_in_collection = collection
                        .iter()
                        .position(|n| ptr::addr_eq(n, node.as_ptr()))
                        .expect("node not found in collection");

                    unsafe {
                        (*collection.nodes.get()).remove(node_index_in_collection);
                        // Clear parent.
                        node.as_ref().h.parent.set(None);
                    }
                }
                EditKind::Modify => {
                    // Handle modification logic here
                }
            }
        }
        self.history.push(history_entry);
    }
}

impl Drop for Document {
    fn drop(&mut self) {
        // TODO: free all nodes in the document tree
    }
}

struct CursorState {
    node: NonNull<Node>,
}

new_key_type! {
    pub struct Cursor;
}

#[derive(Clone, Copy)]
enum EditKind {
    Insert { node: NonNull<Node>, index_in_collection: usize },
    Remove { node: NonNull<Node>, index_in_collection: Option<usize> },
    // TODO
    Modify,
}

/// Represents a single change to the document tree.
#[derive(Clone, Copy)]
struct Edit {
    /// The node that was modified.
    node: *const Node,
    /// The index of the field that was modified.
    field: usize,
    /// The nature of the modification.
    kind: EditKind,
}

struct HistoryEntry {
    changes: Vec<Edit>,
}

/// Represents a value in a node property.
#[derive(Clone, PartialEq)]
pub enum Value {
    String(String),
    Number(f64),
    Boolean(bool),
}

/// Collection of references to nodes.
pub struct NodeRefCollection {
    /// NOTE: This is mutated only when applying edits.
    nodes: UnsafeCell<Vec<NonNull<Node>>>,
}

impl NodeRefCollection {
    pub fn new() -> Self {
        NodeRefCollection { nodes: UnsafeCell::new(Vec::new()) }
    }
}

/// Holds a collection of child nodes.
pub struct NodeCollection {
    // NOTE: The data inside the UnsafeCell is always safe to make a shared reference to:
    //       - through `&mut NodeCollection`, we know that we have exclusive access, so we can just use get_mut
    //       - through `&NodeCollection`: the `&NodeCollection` derives necessarily from `&Document`
    //         which ensures that the document (and all node data inside it) stays immutable
    //         as long as the `&NodeCollection` reference is alive.
    //         And, by contract, document data is only modified inside `apply_pending_changes`,
    //         which requires a mutable reference to the document.
    nodes: UnsafeCell<Vec<NonNull<Node>>>,
}

impl NodeCollection {
    pub fn new() -> Self {
        NodeCollection { nodes: UnsafeCell::new(Vec::new()) }
    }

    /*/// Inserts a node into the collection.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no iterator to the collection is alive while this method is
    /// called.
    unsafe fn insert(&self, parent: &Node, field_index_in_parent: usize, node: &Node, index: usize) {}

    /// Retains only the nodes in the collection for which the predicate returns true.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no iterator to the collection is alive while this method is
    /// called.
    unsafe fn retain<'a>(&'a self, mut f: impl FnMut(&'a Node) -> bool) {
        // SAFETY: the references to the nodes are valid for the lifetime of the collection.
        unsafe {
            (*self.nodes.get()).retain(|&ptr| {
                match f(ptr.as_ref()) {
                    true => true,
                    false => {
                        // Clear parent.
                        ptr.as_ref().h.parent.set(None);
                        false
                    }
                }
            })
        }
    }*/

    /// Iterates over the nodes in the collection.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a Node> + 'a {
        // SAFETY: the references to the nodes are valid for the lifetime of the collection.
        unsafe { (*self.nodes.get()).iter().map(|&ptr| ptr.as_ref()) }
    }
}

impl Index<usize> for NodeCollection {
    type Output = Node;

    fn index(&self, index: usize) -> &Self::Output {
        // SAFETY: the reference to the node is valid for the lifetime of the collection.
        // SAFETY: w.r.t. reference to UnsafeCell aliasing rules: `&NodeCollection` ultimately borrows from `&Document`,
        //         so a `&mut Document` reference cannot be formed while this reference is alive,
        //         so the data inside UnsafeCell will stay immutable for the lifetime of this reference.
        //         (by contract, the only place that mutates the collection is
        //          `apply_pending_changes`, which requires a mutable reference to the document)
        unsafe { (&(*self.nodes.get()))[index].as_ref() }
    }
}

/// Represents a field of a node.
///
/// Each instance corresponds to a field in the declaration associated to a node.
pub enum NodeField {
    // All variants are options because the representation is meant for editing,
    // and when a node is just inserted the "no value" state makes sense.
    // It would not represent a valid document tree, but it is a valid editing state.
    /// A primitive value.
    ///
    /// Corresponds to a field of type [`Type::Primitive`].
    Value(RefCell<Option<Value>>),
    /// A child node.
    ///
    /// Corresponds to a field of type [`Type::NodeInstance`] with multiplicity `One` or `Optional`.
    Node(Cell<Option<NonNull<Node>>>),
    /// A collection of child nodes.
    ///
    /// Corresponds to a field of type [`Type::NodeInstance`] with multiplicity `Many` or `Many1`.
    Collection(NodeCollection),
    // TODO
    RefNode(Cell<Option<NonNull<Node>>>),
    RefCollection(NodeRefCollection),
}

impl NodeField {
    pub fn as_value(&self) -> Option<&RefCell<Option<Value>>> {
        match self {
            NodeField::Value(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_node(&self) -> Option<&Node> {
        match self {
            NodeField::Node(n) => Some(n.get().map(|p| unsafe { p.as_ref() }).unwrap()),
            _ => None,
        }
    }

    pub fn as_collection(&self) -> Option<&NodeCollection> {
        match self {
            NodeField::Collection(c) => Some(c),
            _ => None,
        }
    }
}

/// Node header.
#[repr(C)]
struct NodeHeader {
    /// Pointer to the parent document.
    document: NonNull<Document>,

    /// Pointer to the parent node, if any + field index in parent.
    ///
    /// As long as this node is alive, then it must have a valid parent.
    parent: Cell<Option<(NonNull<Node>, usize)>>,

    /// Node declaration (its "type").
    decl: &'static NodeDecl<'static>,
}

/// Represents a node in the document tree.
///
/// Access the parent node via [`parent`](Node::parent), and access the fields via [`field`](Node::field).
///
/// # Delayed modifications
///
/// Methods such as [`insert`](Node::insert), [`insert_in`](Node::insert_in), and [`remove_self`](Node::remove_self) do not modify
/// the document immediately, and instead store modifications in the document's _pending changes_ list.
///
/// To apply the modifications, call [`apply_pending_changes`](Document::apply_pending_changes).
/// Until then, nodes reflect their state before the modifications.
///
/// # Internals
///
/// `Node` is a dynamically-sized type (DST) composed of a [node header](NodeHeader) followed by an
/// array of [node children](NodeField), holding the data for the node's fields.
/// We allocate both the node and its fields in a single allocation, which is possible since
/// the number of fields is defined by the [NodeDecl] and never changes during the lifetime of the node.
#[repr(C)]
pub struct Node {
    h: NodeHeader,
    children: [NodeField],
}

impl Node {
    /// Returns a pointer to the document that owns this node.
    pub fn document(&self) -> &Document {
        // SAFETY: the document pointer is valid for the lifetime of this node.
        unsafe { self.h.document.as_ref() }
    }

    fn alloc_layout(count: usize) -> (Layout, usize) {
        let (layout, array_offset) =
            Layout::new::<NodeHeader>().extend(Layout::array::<NodeField>(count).unwrap()).unwrap();
        (layout.pad_to_align(), array_offset)
    }

    /// Returns a pointer to the parent of this node.
    pub fn parent(&self) -> Option<&Node> {
        self.h.parent.get().map(|(p, _index)| unsafe { p.as_ref() })
    }

    /// Inserts a child node into a collection field of this node.
    ///
    /// The actual insertion is delayed until the next call to [`apply_pending_changes`](Document::apply_pending_changes).
    /// Until then, the document tree remains unchanged and will not reflect the modification.
    ///
    /// # Arguments
    /// - `field_index`: The index of the field in this node's declaration.
    ///
    /// # Panics
    /// - If the node being inserted belongs to a different document than this node.
    /// - If the field at `field_index` is not a collection field.
    /// - If the node being inserted is not compatible with the type of the collection field.
    pub fn insert(&self, field_index: usize, node: &Node, index_in_collection: usize) {
        assert!(ptr::addr_eq(node.document(), self.document()), "cannot insert node from a different document");
        self.document().pending_changes.borrow_mut().push(Edit {
            node: self as *const Node,
            field: field_index,
            kind: EditKind::Insert { node: NonNull::from_ref(node), index_in_collection },
        });
    }

    /// Inserts a child node into a collection field of this node.
    ///
    /// The actual insertion is delayed until the next call to [`apply_pending_changes`](Document::apply_pending_changes).
    /// Until then, the document tree remains unchanged and will not reflect the modification.
    ///
    /// # Arguments
    /// - `field`: The field declaration of the collection field in this node's declaration.
    /// - `node`: The node to insert into the collection.
    /// - `index_in_collection`: The index at which to insert the node in the collection.
    ///
    /// # Panics
    /// - If the field does not belong to this node's declaration.
    /// - If the node being inserted belongs to a different document than this node.
    /// - If the field is not a collection field.
    pub fn insert_in(&self, field: &FieldDecl, node: &Node, index_in_collection: usize) {
        assert_eq!(field.parent as *const NodeDecl, self.h.decl as *const NodeDecl);
        self.insert(field.index, node, index_in_collection);
    }

    /// Removes a child node from a collection field of this node.
    ///
    /// The actual removal is delayed until the next call to [`apply_pending_changes`](Document::apply_pending_changes).
    /// Until then, the document tree remains unchanged and will not reflect the modification.
    ///
    pub fn remove_in(&self, field: &FieldDecl, index_in_collection: usize) {
        assert_eq!(field.parent as *const NodeDecl, self.h.decl as *const NodeDecl);
        let collection = self.children[field.index].as_collection().expect("field is not a collection");
        assert!(index_in_collection < collection.iter().count(), "index out of bounds in collection");
        collection[index_in_collection].remove_self();
    }

    /// Removes this node from the document.
    ///
    /// The actual removal is delayed until the next call to [`apply_pending_changes`](Document::apply_pending_changes).
    /// Until then, the document tree remains unchanged and will not reflect the modification.
    ///
    /// # Panics
    ///
    /// - When attempting to remove the root node of the document.
    pub fn remove_self(&self) {
        let (parent, field_index) = self.h.parent.get().expect("cannot remove root node");
        self.document().pending_changes.borrow_mut().push(Edit {
            node: parent.as_ptr(),
            field: field_index,
            kind: EditKind::Remove { node: NonNull::from_ref(self), index_in_collection: None },
        });
    }

    /// Creates a new node.
    ///
    /// This will allocate a new node on the heap with an array of fields as specified in [`NodeDecl::fields`].
    ///
    /// # Arguments
    /// - `document`: parent document.
    /// - `decl`: node declaration.
    ///
    pub(crate) fn new(document: &Document, decl: &'static NodeDecl<'static>) -> *mut Node {
        // We can't construct NodeData directly since it's a DST. Instead, we construct
        // a `NodeDataInit<[T; N]>`, which is then unsize-coerced to `NodeDataInit<[T]>`, which we
        // can then reinterpret as `NodeData` since they have the same layout.

        let n = decl.fields.len();
        let (layout, child_offset) = Node::alloc_layout(n);
        let node = unsafe {
            // Allocate header + children array.
            let ptr = alloc(layout);
            let child_ptr = ptr.add(child_offset) as *mut NodeField;

            // Write the header.
            ptr::write(
                ptr as *mut NodeHeader,
                NodeHeader { document: NonNull::from_ref(document), parent: Cell::new(None), decl },
            );

            // Initialize children array to default values.
            for i in 0..n {
                let init = match (decl.fields[i].ty, decl.fields[i].multiplicity) {
                    (Type::Primitive(_), _) => NodeField::Value(RefCell::new(None)),
                    (Type::NodeInstance(_), Multiplicity::One | Multiplicity::Optional) => {
                        NodeField::Node(Cell::new(None))
                    }
                    (Type::NodeInstance(_), Multiplicity::Many | Multiplicity::Many1) => {
                        NodeField::Collection(NodeCollection::new())
                    }
                    (Type::NodeRef(_), Multiplicity::One | Multiplicity::Optional) => {
                        NodeField::RefNode(Cell::new(None))
                    }
                    (Type::NodeRef(_), Multiplicity::Many | Multiplicity::Many1) => {
                        NodeField::RefCollection(NodeRefCollection::new())
                    }
                };
                ptr::write(child_ptr.add(i), init);
            }

            // Cast it to a dummy slice of `[NodeDataHeader]` to set the length metadata.
            // The resulting pointer is not valid for such an array, but it will never be read from,
            // and is only there to build a fat pointer with length metadata.
            let ptr = ptr::slice_from_raw_parts_mut(ptr, n);
            // Now cast it to the final type, and wrap in Box.
            let ptr = ptr as *mut Node;
            // SAFETY: ptr points to memory allocated with the global allocator, and its layout
            //         is valid for an instance NodeData.
            ptr
        };

        node
    }

    /// Returns the layout description of this node, if any.
    pub fn layout(&self) -> Option<&LayoutItem<'_>> {
        self.h.decl.layout
    }

    /// Returns the child at the given index, if any.
    pub fn field(&self, index: usize) -> Option<&NodeField> {
        self.children.get(index)
    }
}

#[cfg(test)]
mod tests {
    use crate::declare_node;

    // dummy language declaration for testing
    declare_node!(
        pub static TEST_NODE: "test_node" {
            name: (str)
            children: ([TEST_CHILD])
        }
    );

    declare_node!(
        pub static TEST_CHILD: "test_child" {
            value: (num)
        }
    );
}
