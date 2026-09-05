use clap::Parser;
use regex::Regex;
use roxmltree::Node;
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io;
use std::io::{BufWriter, LineWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

// Implementation notes:
// * We unwrap liberally, error handling isn't worth it.
// * Parsing vk.xml is complicated, because a lot of information for generating FFI bindings is
//   not defined in XML nodes and attributes, but rather in bits of C source code that we have to parse.
//   (e.g. constant values can be defined as `(~0U)`, which is C syntax). There's no specification on
//   what subset of C is allowed in vk.xml, and the expectation seems to be that those bits will be
//   fed to a C compiler anyway. Some relevant issues:
//   - https://github.com/KhronosGroup/Vulkan-Docs/issues/2596
//   - https://github.com/KhronosGroup/Vulkan-Docs/issues/931
//   - https://github.com/KhronosGroup/Vulkan-Docs/issues/210
//   It seems vk.xml wasn't designed with FFI bindings in mind, which raises the question of
//   why it was created at all, if it's only to generate the C headers. At this point it might be
//   easier to parse the C headers directly instead of decoding whatever crap is in vk.xml.
// * There is absolutely no guarantee that this program will run correctly on new versions of vk.xml
//   without changes.

static VK_XML: &str = include_str!("vk.xml");

#[derive(Parser)]
struct Cli {
    /// Output directory (crate root).
    output_dir: PathBuf,
}

fn main() {
    let args = Cli::parse();
    let xml = roxmltree::Document::parse(VK_XML).unwrap();
    generate(&args.output_dir, &xml).unwrap();
}

fn generate(out_dir: &Path, document: &roxmltree::Document) -> io::Result<()> {
    let root = document.root();
    let registry = child_tagged(root, "registry").unwrap();

    // map enum_name -> enum type
    // necessary for <enum> entries in vk.xml which have only an alias, with no way to deduce the
    // type
    let mut enum_types = EnumTypeMap::new();
    //let mut type_map = TypeMap::new();

    // src/vk.rs
    {
        let out_file = File::create(out_dir.join("src/vk.rs"))?;
        let mut buf_writer = BufWriter::new(out_file);
        let mut line_writer = LineWriter::new(IndentedWriter::new(&mut buf_writer));
        // PAIN: <type> definitions don't say if they belong to the vulkan API (as opposed to, say,
        //       being VulkanSC only). However, this information is present in the <extension> blocks,
        //       which only *reference* the types.
        //       So we have to generate the types as we parse the features and extensions,
        //       resolving the type references into the <types> element.
        //       However, hilariously, <type> nodes have three different ways of defining their names
        //       (either through a "name" attribute, or in different child elements for structs and
        //       function pointers), so we build this "type map" beforehand to accelerate the lookup.
        //       I don't see why the api information couldn't be specified in the <type> element.
        let mut tymap = parse_types(registry);
        let mut cmdmap = parse_commands(registry);
        gen_preamble(&mut line_writer)?;
        gen_enums(&mut line_writer, registry, &mut enum_types)?;
        gen_features(&mut line_writer, registry, &mut enum_types, &mut tymap, &mut cmdmap)?;
        gen_extensions(&mut line_writer, registry, &mut enum_types, &mut tymap, &mut cmdmap)?;
        gen_core_dispatch_tables(&mut line_writer, registry, &cmdmap)?;
    }
    Ok(())
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum DispatchType {
    Entry,
    Instance,
    Device,
}
type EnumTypeMap = HashMap<String, String>;
#[derive(Copy, Clone, Eq, PartialEq)]
enum HandleType {
    Dispatchable,
    NonDispatchable,
}
#[derive(Clone)]
enum Category {
    Enum,
    Bitmask(String),
    Handle(HandleType),
    FuncPointer(FuncInfo),
    Struct,
    Union,
    Trash,
}
#[derive(Clone)]
struct TypeInfo<'a, 'input> {
    node: Node<'a, 'input>,
    name: String,
    alias: Option<String>,
    category: Category,
    generated: Cell<bool>,
}
type TypeMap<'a, 'input> = HashMap<String, TypeInfo<'a, 'input>>;

#[derive(Clone, Default)]
struct FuncInfo {
    proto: CDecl,
    params: Vec<CDecl>,
}

struct CommandInfo<'a, 'input> {
    node: Node<'a, 'input>,
    name: String,
    alias: Option<String>,
    func: FuncInfo,
    dispatch: DispatchType,
    generated: Cell<bool>,
}
type CommandMap<'a, 'input> = HashMap<String, CommandInfo<'a, 'input>>;

static PREAMBLE: &str = r#"use crate::macros::*;   // handle, nondispatchable_handle
use crate::platform_types::*;
use crate::video::*;
use std::ffi::*;
use std::ptr;

pub type VkSampleMask = u32;
pub type VkBool32 = u32;
pub type VkFlags = u32;
pub type VkFlags64 = u64;
pub type VkDeviceSize = u64;
pub type VkDeviceAddress = u64;
"#;

fn gen_preamble(out: &mut Writer) -> io::Result<()> {
    writeln!(out, "{PREAMBLE}\n")?;
    Ok(())
}

/*
fn collect_vulkan_apis(registry: Node) -> HashSet<String> {
    let mut vulkan_apis = HashSet::new();
    for ext in children_tagged(child_tagged(registry, "extensions").unwrap(), "extension") {
        if is_vulkan_supported_extension(ext) {
            for require in children_tagged(ext, "require") {
                for ty in children_tagged(require, "type") {
                    if let Some(name) = ty.attribute("name") {
                        vulkan_apis.insert(name.to_string());
                    }
                }
            }
        }
    }
    vulkan_apis
}*/

/*
fn gen_type_map<'a, 'input>(registry: Node<'a, 'input>) -> io::Result<TypeMap<'a, 'input>> {
    let mut type_map = TypeMap::new();
    for types in children_tagged(registry, "types") {
        for node in children_tagged(types, "type") {
            if !api_check(node) {
                continue;
            }

            let category = match node.attribute("category") {
                Some("enum") => Category::Enum,
                Some("bitmask") => Category::Bitmask,
                Some("handle") => Category::Handle,
                Some("funcpointer") => Category::FuncPointer,
                Some("struct") => Category::Struct,
                Some("union") => Category::Union,
                _ => Category::Trash, // whatever
            };
            type_map.insert(name.to_string(), TypeInfo { node, category, generated: false });
        }
    }
    Ok(type_map)
}*/

fn parse_types<'a, 'input>(registry: Node<'a, 'input>) -> TypeMap<'a, 'input> {
    let mut type_map = TypeMap::new();
    for types in children_tagged(registry, "types") {
        for node in children_tagged(types, "type") {
            if !api_check(node) {
                continue;
            }
            if let Some(tyinfo) = parse_type(node, &type_map) {
                type_map.insert(tyinfo.name.clone(), tyinfo);
            }
        }
    }
    type_map
}

fn parse_type<'a, 'input>(node: Node<'a, 'input>, tymap: &TypeMap<'a, 'input>) -> Option<TypeInfo<'a, 'input>> {
    let Some(category) = node.attribute("category") else {
        return None;
    };
    let alias = node.attribute("alias").map(sanitize_ident).map(|s| s.to_string());

    let name;
    let cat;

    match category {
        "funcpointer" => {
            let func_info = parse_command_or_funcptr(node).unwrap();
            name = func_info.proto.name.clone();
            cat = Category::FuncPointer(func_info);
        }
        "enum" => {
            // PAIN: Enum <type> records don't actually specify their type, so we can't generate them here.
            name = node.attribute("name").unwrap().to_string();
            cat = Category::Enum;
        }
        "bitmask" => {
            name = node
                .attribute("name")
                .or_else(|| child_tagged(node, "name").map(|n| n.text().unwrap()))
                .unwrap()
                .to_string();
            let ty = child_tagged(node, "type").map(|n| n.text().unwrap()).or_else(|| alias.as_deref()).unwrap();
            let rust_ty = c_type_to_rust(ty, false).to_string();
            cat = Category::Bitmask(rust_ty);
        }
        "handle" => {
            // For handles
            name = child_tagged(node, "name")
                .map(|n| n.text().unwrap())
                .or_else(|| node.attribute("name"))
                .unwrap()
                .to_string();
            let ty = child_tagged(node, "type").map(|n| n.text().unwrap()).or_else(|| alias.as_deref()).unwrap();
            let handle_type = match ty {
                "VK_DEFINE_HANDLE" => {
                    HandleType::Dispatchable
                    //write!(out, "handle!({name}, {name}_T);\n")?;
                }
                "VK_DEFINE_NON_DISPATCHABLE_HANDLE" => {
                    HandleType::NonDispatchable
                    //write!(out, "non_dispatchable_handle!({name});\n")?;
                }
                _ => {
                    let alias = tymap.get(ty).unwrap();
                    match alias.category {
                        Category::Handle(handle_type) => handle_type,
                        _ => panic!("unexpected handle type"),
                    }
                }
            };
            cat = Category::Handle(handle_type);
        }
        "struct" => {
            name = node.attribute("name").unwrap().to_string();
            cat = Category::Struct;
        }
        "union" => {
            name = node.attribute("name").unwrap().to_string();
            cat = Category::Union;
        }
        _ => {
            // Trash that we don't want to parse, like basetypes, etc.
            //
            // PAIN: "basetype" entries (e.g. `VkBool32`, `VkFlags`, etc.) are not generated there because
            //       generating them from vk.xml is intractable in general (some of them contain preprocessor directives...)
            // PAIN:
            // For bitmasks, the name is in a child tag
            //      unless it's an alias.
            // For enums, the name is in the "name" attribute.
            // For handles, it's usually in a child tag.
            // For <funcpointer>, it's within a <name> tag inside <proto>
            name = node
                .attribute("name")
                .or_else(|| child_tagged(node, "name").map(|n| n.text().unwrap()))
                .or_else(|| {
                    child_tagged(node, "proto").and_then(|p| child_tagged(p, "name")).map(|n| n.text().unwrap())
                })
                .unwrap()
                .to_string();
            cat = Category::Trash;
        }
    }
    Some(TypeInfo { node, name, alias, category: cat, generated: Cell::new(false) })
}

fn gen_type(out: &mut Writer, tyinfo: &TypeInfo, tymap: &TypeMap) -> io::Result<()> {
    if let Some(ref alias) = tyinfo.alias {
        writeln!(out, "pub type {} = {};", tyinfo.name, alias)?;
        return Ok(());
    }
    match &tyinfo.category {
        Category::Enum => {
            // PAIN: Enum <type> records don't actually specify their type, so we can't generate them here.
        }
        Category::Bitmask(rust_ty) => {
            writeln!(out, "pub type {} = {};", tyinfo.name, rust_ty)?;
        }
        Category::Handle(handle_type) => {
            let name = &tyinfo.name;
            match handle_type {
                HandleType::Dispatchable => writeln!(out, "handle!({name}, {name}_T);")?,
                HandleType::NonDispatchable => writeln!(out, "non_dispatchable_handle!({name});")?,
            }
        }
        Category::FuncPointer(func_info) => {
            write!(out, "pub type {} = Option<unsafe extern \"system\" fn", tyinfo.name)?;
            gen_func_sig(out, func_info)?;
            writeln!(out, ">;")?;
        }
        Category::Struct | Category::Union => {
            let name = &tyinfo.name;
            writeln!(out, "/// <https://docs.vulkan.org/refpages/latest/refpages/source/{name}.html>")?;
            writeln!(out, "#[repr(C)]")?;
            writeln!(out, "#[cfg_attr(feature = \"debug\", derive(Debug))]")?;
            writeln!(out, "#[derive(Copy, Clone)]")?;
            let is_union = matches!(tyinfo.category, Category::Union);
            let kind = match tyinfo.category {
                Category::Struct => "struct",
                Category::Union => "union",
                _ => unreachable!(),
            };
            write!(out, "pub {kind} {name} {{\n")?;
            indent(out);
            for member in children_tagged(tyinfo.node, "member") {
                if !api_check(member) {
                    continue;
                }
                // PAIN: The text of <member> is actually C syntax for a struct member.
                //       There is markup for the <name> and <type> but the cv-qualifiers, pointer & array declarators
                //       are not marked up, so they are basically useless and we ignore them.
                //       Parse the C syntax directly instead.
                let values = member.attribute("values");
                let optional = member.attribute("optional").unwrap_or("false").split(',').collect::<Vec<_>>();
                let last_opt = optional.last().map(|&s| s == "true").unwrap_or(false);
                let len = member.attribute("len");
                write!(out, "pub ")?;
                let text = node_text(&member);
                let decl = parse_c_declarator(&text).unwrap();
                let name = sanitize_ident(decl.name.as_str());
                let ty = decl.rust_type(false);
                write!(out, "{name}: {ty}")?;
                // Write default field values.
                if !is_union {
                    match decl.name.as_str() {
                        // sType field
                        "sType" if values.is_some() => write!(out, " = {}", values.unwrap())?,
                        // pNext is always defaultable to null
                        "pNext" => {
                            write!(out, " = {}", if decl.is_const_ptr() { "ptr::null()" } else { "ptr::null_mut()" })?
                        }
                        // pointers with len attributes are defaultable to null (if len == 0)
                        _ if len.is_some() && decl.is_ptr() => {
                            write!(out, " = {}", if decl.is_const_ptr() { "ptr::null()" } else { "ptr::null_mut()" })?
                        }
                        // otherwise, decide based on the "optional" field and whether the field is a
                        // pointer, and also some heuristics
                        _other if last_opt => {
                            if decl.is_const_ptr() {
                                write!(out, " = ptr::null()")?;
                            } else if decl.is_ptr() {
                                write!(out, " = ptr::null_mut()")?;
                            } else if let Some(info) = tymap.get(&decl.inner_ty) {
                                match info.category {
                                    Category::Handle(_) => {
                                        let ty = decl.inner_ty;
                                        write!(out, " = {ty}::null()")?;
                                    }
                                    Category::FuncPointer(_) => {
                                        // This is a function pointer, modeled as `Option<fn()>` in rust.
                                        // The NULL pointer is `None`.
                                        write!(out, " = None")?;
                                    }
                                    Category::Bitmask(_) | Category::Enum => {
                                        // Bitmasks and enums are just integers, so we can default to 0.
                                        write!(out, " = 0")?;
                                    }
                                    _ => {}
                                }
                            } else {
                                // heuristics
                                match decl.inner_ty.as_str() {
                                    "LPCWSTR" => write!(out, " = ptr::null()")?, // *const u16
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
                writeln!(out, ",")?;
            }
            dedent(out);
            writeln!(out, "}}")?;
            // Structs may contain raw pointers and thus not automatically Send/Sync.
            // Rust questionably makes all pointers !Send+!Sync by default to avoid "footguns"
            // but the real unsafety is in the use (via derefs) of the pointers.
            //
            // For convenience, mark all Vulkan structs as Send+Sync.
            // This doesn't make the crate any less sound, it just moves the thread-safety
            // contract to each command's "unsafe" contract.
            writeln!(out, "unsafe impl Send for {name} {{}}")?;
            writeln!(out, "unsafe impl Sync for {name} {{}}")?;
        }
        _ => {}
    }
    Ok(())
}

fn gen_enums(out: &mut Writer, registry: Node, enum_types: &mut EnumTypeMap) -> io::Result<()> {
    for enums in children_tagged(registry, "enums") {
        gen_enum(out, enums, enum_types)?;
    }
    Ok(())
}

fn gen_enum(out: &mut Writer, node: Node, enum_types: &mut EnumTypeMap) -> io::Result<()> {
    enum Kind {
        Bitmask,
        Constants,
        Enum,
    }
    let bitwidth = int_attr(node, "bitwidth").unwrap_or(32);
    let kind = match node.attribute("type").unwrap() {
        "bitmask" => Kind::Bitmask,
        "constants" => Kind::Constants,
        "enum" => Kind::Enum,
        _ => panic!("unexpected enum kind"),
    };
    let ty_name = node.attribute("name").map(sanitize_ident);
    let ty_rust = enum_bitwidth_to_rust(bitwidth);
    // Write bitmask or enum type alias
    match kind {
        Kind::Bitmask | Kind::Enum => {
            let ty_name = ty_name.as_ref().unwrap();
            write!(out, "pub type {ty_name} = {ty_rust};\n")?;
        }
        _ => {}
    }
    for en in children_tagged(node, "enum") {
        let name = sanitize_ident(en.attribute("name").unwrap());
        let bitpos = int_attr(en, "bitpos");
        let value = en.attribute("value").map(c_constant_to_rust);
        let alias = en.attribute("alias");
        maybe_write_deprecated_attr(out, en.attribute("deprecated"), alias)?;
        write!(out, "pub const {}: ", name)?;
        let ty = en.attribute("type").map(|ty| c_type_to_rust(ty, false)).unwrap_or(ty_name.as_ref().unwrap().as_ref());
        write!(out, "{ty}")?;
        write!(out, " = ")?;
        if let Some(bitpos) = bitpos {
            write!(out, "{:#x}", 1u64 << bitpos)?;
        } else if let Some(value) = value {
            write!(out, "{value}")?;
        } else if let Some(alias) = alias {
            write!(out, "{alias}")?;
        };
        writeln!(out, ";")?;
        enum_types.insert(name.to_string(), ty.to_string());
    }
    Ok(())
}

fn parse_command_or_funcptr<'a, 'input>(node: Node<'a, 'input>) -> Option<FuncInfo> {
    let proto = child_tagged(node, "proto").unwrap();
    let proto = node_text(&proto);
    let proto = parse_c_declarator(&proto).unwrap();
    let mut params = vec![];
    for param in children_tagged(node, "param") {
        let text = node_text(&param);
        let decl = parse_c_declarator(&text).unwrap();
        params.push(decl);
    }
    Some(FuncInfo { proto, params })
}

fn parse_command<'a, 'input>(
    node: Node<'a, 'input>,
    command_infos: &CommandMap<'a, 'input>,
) -> Option<CommandInfo<'a, 'input>> {
    if let Some(alias) = node.attribute("alias") {
        let name = node.attribute("name").unwrap();
        let alias_info = command_infos.get(alias).unwrap().clone();
        Some(CommandInfo {
            node,
            name: name.to_string(),
            alias: Some(alias.to_string()),
            func: alias_info.func.clone(),
            dispatch: alias_info.dispatch,
            generated: Cell::new(false),
        })
    } else {
        let func_info = parse_command_or_funcptr(node).unwrap();
        let dispatch = dispatch_type_from_first_param(func_info.params.first().map(|p| p.name.as_str()));
        Some(CommandInfo {
            node,
            name: func_info.proto.name.clone(),
            alias: None,
            func: func_info,
            dispatch,
            generated: Cell::new(false),
        })
    }
}

fn parse_commands<'a, 'input>(registry: Node<'a, 'input>) -> CommandMap<'a, 'input> {
    let mut command_map = CommandMap::new();
    let commands = child_tagged(registry, "commands").unwrap();
    for cmd in children_tagged(commands, "command") {
        if !api_check(cmd) {
            continue;
        }
        if let Some(info) = parse_command(cmd, &command_map) {
            command_map.insert(info.name.clone(), info);
        }
    }
    command_map
}

fn gen_func_sig(out: &mut Writer, func_info: &FuncInfo) -> io::Result<()> {
    write!(out, "(")?;
    for (i, param) in func_info.params.iter().enumerate() {
        if i > 0 {
            write!(out, ", ")?;
        }
        write!(out, "{}: {}", sanitize_ident(&param.name), param.rust_type(false))?;
    }
    write!(out, ") -> {}", func_info.proto.rust_type(true))?;
    Ok(())
}

fn gen_command_pfn(out: &mut Writer, cmd: &CommandInfo) -> io::Result<()> {
    let name = &cmd.name;
    if let Some(ref alias) = cmd.alias {
        write!(out, "pub type PFN_{name} = PFN_{alias};\n")?;
        return Ok(());
    }
    write!(out, "pub type PFN_{name} = unsafe extern \"system\" fn")?;
    gen_func_sig(out, &cmd.func)?;
    writeln!(out, ";")?;
    Ok(())
}

/*
fn gen_funcpointer_ty(out: &mut Writer, func_info: &FuncInfo) -> io::Result<()> {
    write!(out, "Option<unsafe extern \"system\" fn")?;
    gen_func_sig(out, func_info)?;
    write!(out, ">")?;
    Ok(())
}*/

fn gen_features(
    out: &mut Writer,
    registry: Node,
    enum_types: &mut EnumTypeMap,
    tymap: &mut TypeMap,
    cmdmap: &mut CommandMap,
) -> io::Result<()> {
    for feature in children_tagged(registry, "feature") {
        if !api_check(feature) {
            continue;
        }
        let name = feature.attribute("name").unwrap();
        //let number = int_attr(feature, "number").unwrap();
        writeln!(out, "// Feature: {name}")?;
        for require in children_tagged(feature, "require") {
            gen_require(out, require, None, enum_types, tymap, cmdmap)?;
        }
    }
    Ok(())
}

fn gen_extensions(
    out: &mut Writer,
    registry: Node,
    enum_types: &mut EnumTypeMap,
    tymap: &mut TypeMap,
    cmdmap: &mut CommandMap,
) -> io::Result<()> {
    let extensions = child_tagged(registry, "extensions").unwrap();
    for ext in children_tagged(extensions, "extension") {
        if !is_vulkan_supported_extension(ext) {
            continue;
        }
        let name = ext.attribute("name").unwrap();
        let number = int_attr(ext, "number").unwrap();
        writeln!(out, "// Extension: {name} ({number})")?;
        for require in children_tagged(ext, "require") {
            gen_require(out, require, Some(number), enum_types, tymap, cmdmap)?;
        }
    }
    Ok(())
}

fn gen_require(
    out: &mut Writer,
    require: Node,
    extnumber: Option<i64>,
    enum_types: &mut EnumTypeMap,
    tymap: &mut TypeMap,
    cmdmap: &mut CommandMap,
) -> io::Result<()> {
    for en in children_tagged(require, "enum") {
        let name = en.attribute("name").unwrap();
        // PAIN: Some extensions redefine the same constant.
        //       Look it up and skip if we've already emitted it.
        if enum_types.contains_key(name) {
            continue;
        }
        let alias = en.attribute("alias");
        let bitpos = int_attr(en, "bitpos");
        let offset = int_attr(en, "offset");
        let value = en.attribute("value").map(c_constant_to_rust);
        let ty = match en.attribute("extends") {
            Some(extends) => extends,
            None => {
                // PAIN: vk.xml has <enum> elements without <extends> (usually VK_*_SPEC_VERSION and VK_*_EXTENSION_NAME),
                //       and thus no known type.
                //       If we could emit `#defines` this wouldn't be a problem, but, again, we're not emitting C bindings
                //       (and we can't have rust infer the type either, as this is not possible in consts).
                //       So we infer the type from the value. For now, assume that it is either a 32-bit integer or a string.
                // PAIN²: there is also stuff like `<enum name="VK_ANDROID_NATIVE_BUFFER_NAME" alias="VK_ANDROID_NATIVE_BUFFER_EXTENSION_NAME" />`
                //        for which it's impossible to infer the type without somehow resolving the alias.
                //        That's why we have `enum_types`.
                match value {
                    Some(ref value) if value.starts_with('"') => "&'static str",
                    Some(_) => "u32",
                    None => {
                        match alias {
                            Some(alias) => {
                                // Look up the alias. Hopefully we've seen it before and know its type.
                                // If not, bail out; someone should fix vk.xml already.
                                enum_types.get(alias).unwrap()
                            }
                            None => {
                                // PAIN^3: sometimes there's no value or alias, e.g.:
                                //         <enum name="VK_LUID_SIZE_KHR" />
                                //         What are we supposed to do with this?
                                //         Assume that this is a duplicate.
                                continue;
                            }
                        }
                    }
                }
            }
        };
        maybe_write_deprecated_attr(out, en.attribute("deprecated"), alias)?;
        write!(out, "pub const {name}: {ty} = ")?;
        if let Some(bitpos) = bitpos {
            write!(out, "{:#x}", 1u64 << bitpos)?;
        } else if let Some(offset) = offset {
            let extnumber = int_attr(en, "extnumber").or(extnumber).unwrap();
            let mut val = ext_enum_value(extnumber, offset);
            if en.attribute("dir").is_some() {
                val = -val;
            }
            write!(out, "{}", val)?;
        } else if let Some(value) = value {
            write!(out, "{value}")?;
        } else if let Some(alias) = alias {
            write!(out, "{alias}")?;
        };
        writeln!(out, ";")?;
        let ty_str = ty.to_string();
        enum_types.insert(name.to_string(), ty_str);
    }
    for ty in children_tagged(require, "type") {
        let name = ty.attribute("name").unwrap();
        eprintln!("generating  {name}");
        let info = tymap.get(name).expect("type not found");
        if info.generated.get() {
            // PAIN: types,commands,etc. can be defined by multiple extensions.
            continue;
        }
        // PAIN: vk.xml contains video extensions which reference types in another header
        //       not described by vk.xml (video.xml).
        //       For now, vk_video/*.h are bindgen-ed separately, I've had enough with vk.xml already.
        gen_type(out, &info, tymap)?;
        info.generated.set(true)
    }
    for cmd in children_tagged(require, "command") {
        let name = cmd.attribute("name").unwrap();
        let info = cmdmap.get(name).expect("command not found");
        if info.generated.get() {
            continue;
        }
        gen_command_pfn(out, info)?;
        info.generated.set(true);
    }
    Ok(())
}

fn gen_vk_dispatch_table(
    out: &mut Writer,
    version: &str,
    dispatch_kind: &str,
    inherits: Option<&str>,
    funcs: Vec<&CommandInfo>,
) -> io::Result<()> {
    if !funcs.is_empty() {
        writeln!(out, "dispatch_table! {{ Vulkan_{version}_{dispatch_kind};")?;
        indent(out);
        if let Some(ver) = inherits {
            writeln!(out, "[vk_{ver}: Vulkan_{ver}_{dispatch_kind}]")?;
        }
        for cmd in funcs.iter() {
            let vk_cmd_name = &cmd.name;
            let cmd_name = vk_cmd_name.strip_prefix("vk").unwrap();
            writeln!(out, "{cmd_name},PFN_{vk_cmd_name},c\"{vk_cmd_name}\";")?;
        }
        dedent(out);
        writeln!(out, "}}")?;
        // write function wrappers
        writeln!(out, "impl Vulkan_{version}_{dispatch_kind} {{")?;
        indent(out);
        for cmd in funcs.iter() {
            let vk_cmd_name = &cmd.name;
            let cmd_name = vk_cmd_name.strip_prefix("vk").unwrap();
            writeln!(out, "#[inline(always)]")?;
            write!(out, "pub unsafe fn {cmd_name}(&self")?;
            for param in cmd.func.params.iter() {
                write!(out, ", {}: {}", sanitize_ident(&param.name), param.rust_type(false))?;
            }
            writeln!(out, ") -> {} {{", cmd.func.proto.rust_type(true))?;
            indent(out);
            write!(out, "(self.{cmd_name})(")?;
            for (i, param) in cmd.func.params.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}", sanitize_ident(&param.name))?;
            }
            writeln!(out, ")")?;
            dedent(out);
            writeln!(out, "}}")?;
        }
        dedent(out);
        writeln!(out, "}}")?;
    }
    Ok(())
}

fn gen_core_dispatch_tables(out: &mut Writer, registry: Node, cmds: &CommandMap) -> io::Result<()> {
    // PAIN: Within vk.xml, Vulkan API versions are divided in smaller features (VK_{BASE,GRAPHICS,COMPUTE}_VERSION_*_*),
    //       and form a dependency tree via the "depends" attribute. However, this is useless for
    //       grouping commands by API version, so rely on "number" instead.
    let mut api_version_features = BTreeMap::new();
    for feature in children_tagged(registry, "feature") {
        let number = feature.attribute("number").unwrap().replace('.', "_");
        api_version_features.entry(number).or_insert_with(Vec::new).push(feature);
    }
    for (api_version, features) in api_version_features {
        // split instance/device commands
        let mut entry_fns = vec![];
        let mut instance_fns = vec![];
        let mut device_fns = vec![];
        for feature in features.iter() {
            for req in children_tagged(*feature, "require") {
                for cmd in children_tagged(req, "command") {
                    let info = &cmds[cmd.attribute("name").unwrap()];
                    if !info.generated.get() {
                        continue;
                    }
                    match &info.dispatch {
                        DispatchType::Entry => entry_fns.push(info),
                        DispatchType::Instance => instance_fns.push(info),
                        DispatchType::Device => device_fns.push(info),
                    }
                }
            }
        }
        // shamelessly hardcode previous version dependencies
        let instance_previous_version = match &*api_version {
            //"1_5" => Some("1_4"), // not there yet!
            "1_3" => Some("1_1"),
            "1_1" => Some("1_0"),
            _ => None,
        };
        let device_previous_version = match &*api_version {
            //"1_5" => Some("1_4"), // not there yet!
            "1_4" => Some("1_3"),
            "1_3" => Some("1_2"),
            "1_2" => Some("1_1"),
            "1_1" => Some("1_0"),
            _ => None,
        };
        writeln!(out, "// Vulkan {api_version}")?;
        gen_vk_dispatch_table(out, &api_version, "EntryDispatch", None, entry_fns)?;
        gen_vk_dispatch_table(out, &api_version, "InstanceDispatch", instance_previous_version, instance_fns)?;
        gen_vk_dispatch_table(out, &api_version, "DeviceDispatch", device_previous_version, device_fns)?;
    }
    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Helpers
//--------------------------------------------------------------------------------------------------

/// Checks if a node has an `api` attribute that contains `vulkan`, or no `api` at all.
///
/// Returns false if the `api` doesn't contain `vulkan`.
fn api_check(node: Node) -> bool {
    node.attribute("api").map(|s| s.split(',').any(|s| s == "vulkan")).unwrap_or(true)
}

/// Returns the command type based on its first parameter.
fn dispatch_type_from_first_param(first_param_type: Option<&str>) -> DispatchType {
    // The overhead of the internal dispatch for VkDevice objects can be avoided by obtaining
    // device-specific function pointers for any commands that use a device or device-child object as their dispatchable object.
    match first_param_type {
        Some("VkInstance") | Some("VkPhysicalDevice") => DispatchType::Instance,
        Some("VkDevice") | Some("VkQueue") | Some("VkCommandBuffer") => DispatchType::Device,
        _ => DispatchType::Entry,
    }
}

/// Writes a `deprecated` attribute.
fn maybe_write_deprecated_attr(out: &mut Writer, deprecated: Option<&str>, alias: Option<&str>) -> io::Result<()> {
    match deprecated {
        Some("true") => write!(out, "#[deprecated]\n")?,
        Some("aliased") => write!(out, "#[deprecated(note = \"use {} instead\")]\n", alias.unwrap())?,
        Some(reason) => write!(out, "#[deprecated(note = \"{reason}\")]\n")?,
        None => {}
    }
    Ok(())
}

/// Info about a C declarator.
#[derive(Clone, Default)]
struct CDecl {
    /// Name (e.g. `name` in `const char* name[10]`)
    name: String,
    inner_ty: String,
    inner_const: bool,
    outer_const: bool,
    array_len_inner: Option<String>,
    array_len_outer: Option<String>,
    inner_ptr: bool,
    outer_ptr: bool,
    bitfield: Option<String>,
}

impl CDecl {
    fn rust_type(&self, return_type: bool) -> String {
        let mut ty = c_type_to_rust(&self.inner_ty, return_type).to_string();
        if self.inner_ptr {
            if self.inner_const {
                ty = format!("*const {ty}");
            } else {
                ty = format!("*mut {ty}");
            }
        }
        if self.outer_ptr {
            if self.outer_const {
                ty = format!("*const {ty}");
            } else {
                ty = format!("*mut {ty}");
            }
        }
        if let Some(ref array_len) = self.array_len_inner {
            ty = format!("[{ty}; {} as usize]", array_len);
        }
        if let Some(ref array_len) = self.array_len_outer {
            ty = format!("[{ty}; {} as usize]", array_len);
        }
        ty
    }

    fn is_const_ptr(&self) -> bool {
        (self.outer_ptr && self.outer_const) || (!self.outer_const && self.inner_ptr && self.inner_const)
    }

    fn is_ptr(&self) -> bool {
        self.outer_ptr || self.inner_ptr
    }
}

/// Converts a C declarator to its Rust equivalent and writes it to `out`.
///
/// Returns parsed information about the declarator.
///
/// # Example
/// `const char* name[10]` -> `name: [*const c_char; 10]`, returns (`name`, false)
/// `struct VkBaseInStructure* pNext` -> `pNext: *mut VkBaseInStructure`, returns (`pNext`, false)
fn parse_c_declarator<'a>(member: &'a str) -> io::Result<CDecl> {
    // [const] [struct] <type> [*|const *] <name> [ [<array_len>] ] [: bitfield]
    // https://regex101.com/?regex=%5E%5Cs*%28const%5Cs%2B%29%3F%28struct%5Cs%2B%29%3F%28%5Cw%2B%29%5Cs*%28%5C*%29%3F%5Cs*%28const%29%3F%5Cs*%28%5C*%29%3F%5Cs*%28%5Cw%2B%29%5Cs*%28%3F%3A%5C%5B%5Cs*%28%5Cw%2B%29%5Cs*%5C%5D%29%3F%24&testString=const++char32*+const*+ppEnabledExtensionNames+%5B4%5D%0Aconst+++struct+++VkBaseInStructure*+pNext%0AVkStructureType+sType%0Auint8_t+pipelineCacheUUID%5BVK_UUID_SIZE%5D%0Avoid**+ppData%0A&flags=gmu&flavor=rust&delimiter=%22
    static DECL_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^\s*(const\s+)?(struct\s+)?(\w+)\s*(\*)?\s*(const)?\s*(\*)?\s*(\w+)\s*(?:\[\s*(\w+)\s*])?\s*(?:\[\s*(\w+)\s*])?\s*(?::\s*(\d+))?\s*$")
            .unwrap()
    });
    // 1 inner const
    // 2 struct
    // 3 inner type
    // 4 inner ptr
    // 5 outer const
    // 6 outer ptr
    // 7 name
    // 8 array len
    // 9 bitfield
    let caps = DECL_RE.captures(member).unwrap();
    let inner_const = caps.get(1).map(|m| m.as_str()).is_some();
    //let struct_ = caps.get(2).map(|m| m.as_str());
    let inner_ty = caps.get(3).map(|m| m.as_str().to_string()).unwrap();
    let inner_ptr = caps.get(4).map(|m| m.as_str()).is_some();
    let outer_const = caps.get(5).map(|m| m.as_str()).is_some();
    let outer_ptr = caps.get(6).map(|m| m.as_str()).is_some();
    let name = caps.get(7).map(|m| m.as_str().to_string()).unwrap();
    let array_len_outer = caps.get(8).map(|m| m.as_str().to_string());
    let array_len_inner = caps.get(9).map(|m| m.as_str().to_string());
    let bitfield = caps.get(10).map(|m| m.as_str().to_string());

    Ok(CDecl {
        name,
        inner_ty,
        inner_const,
        outer_const,
        array_len_inner,
        array_len_outer,
        inner_ptr,
        outer_ptr,
        bitfield,
    })
}

/// Crude parser for C-style constants in vk.xml.
fn c_constant_to_rust(mut value: &str) -> String {
    match value {
        "(~0U)" => "u32::MAX".to_string(),
        "(~1U)" => "u32::MAX - 1".to_string(), // VK_QUEUE_FAMILY_EXTERNAL
        "(~2U)" => "u32::MAX - 2".to_string(), // VK_QUEUE_FAMILY_FOREIGN_EXT
        "(~0ULL)" => "u64::MAX".to_string(),
        other => {
            let hex = value.starts_with("0x") || value.starts_with("-0x");
            if let Some(value) = value.strip_suffix("f")
                && !hex
            {
                format!("{}f32", value)
            } else if let Some(value) = value.strip_suffix("F")
                && !hex
            {
                format!("{}f32", value)
            } else if let Some(value) = value.strip_suffix("U") {
                format!("{}u32", value)
            } else if let Some(value) = value.strip_suffix("UL") {
                format!("{}u32", value)
            } else if let Some(value) = value.strip_suffix("ULL") {
                format!("{}u64", value)
            } else {
                value.to_string()
            }
        }
    }
}

/// Converts a C platform type to its rust equivalent
fn c_type_to_rust(ty: &str, return_type: bool) -> &str {
    match ty {
        "uint8_t" => "u8",
        "uint16_t" => "u16",
        "uint32_t" => "u32",
        "uint64_t" => "u64",
        "int8_t" => "i8",
        "int16_t" => "i16",
        "int32_t" => "i32",
        "int64_t" => "i64",
        "float" => "f32",
        "double" => "f64",
        "size_t" => "usize",
        "intptr_t" => "isize",
        "uintptr_t" => "usize",
        "void" if return_type => "()",
        "void" if !return_type => "c_void",
        "char" => "c_char",
        "int" => "c_int",
        other => other,
    }
}

fn enum_bitwidth_to_rust(bitwidth: i64) -> &'static str {
    match bitwidth {
        32 => "i32",
        64 => "i64",
        _ => panic!("unexpected bitwidth"),
    }
}

fn sanitize_ident(ident: &str) -> Cow<str> {
    match ident {
        "type" | "mod" | "ref" | "self" | "super" | "crate" => Cow::Owned(format!("r#{}", ident)),
        _ => Cow::Borrowed(ident),
    }
}

/// Returns an enumerant value given its extension number and offset.
///
/// See https://registry.khronos.org/vulkan/specs/latest/styleguide.html#extensions-assigning-token-values
fn ext_enum_value(extnum: i64, offset: i64) -> i64 {
    let base = 1_000_000_000;
    let range = 1_000;
    base + (extnum - 1) * range + offset
}

fn is_vulkan_supported_extension(extnode: Node) -> bool {
    extnode.attribute("supported").map(|s| s.split(',').any(|s| s == "vulkan")).unwrap_or(false)
}

struct IndentedWriter<'a> {
    indent: usize,
    inner: &'a mut dyn Write,
}

impl<'a> IndentedWriter<'a> {
    fn new(inner: &'a mut dyn Write) -> Self {
        Self { indent: 0, inner }
    }
}

impl<'a> Write for IndentedWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        write!(self.inner, "{:indent$}", "", indent = self.indent)?;
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

type Writer<'a> = LineWriter<IndentedWriter<'a>>;

const INDENT: usize = 4;

/// Increase indentation on the specified writer.
fn indent(out: &mut Writer) {
    out.get_mut().indent += INDENT;
}

fn dedent(out: &mut Writer) {
    out.get_mut().indent -= INDENT;
}

fn child_tagged<'a, 'input>(node: Node<'a, 'input>, tag: &str) -> Option<Node<'a, 'input>> {
    node.children().find(|n| n.tag_name().name() == tag)
}

fn children_tagged<'a, 'input, 'tag>(
    node: Node<'a, 'input>,
    tag: &'tag str,
) -> impl Iterator<Item = Node<'a, 'input>> + 'tag
where
    'input: 'tag,
    'a: 'tag,
{
    node.children().filter(move |n| n.tag_name().name() == tag)
}

fn child_by_name_attribute<'a, 'input>(node: Node<'a, 'input>, name: &str) -> Option<Node<'a, 'input>> {
    node.children().find(|n| n.attribute("name") == Some(name))
}

fn node_text(node: &Node) -> String {
    node.children()
        .filter_map(|n| {
            // filter out <comment>, which appear directly inside <member> ...
            if n.tag_name().name() != "comment" { n.text() } else { None }
        })
        .collect::<String>()
}

fn int_attr(node: Node, name: &str) -> Option<i64> {
    node.attribute(name).map(|s| s.parse::<i64>().unwrap())
}
