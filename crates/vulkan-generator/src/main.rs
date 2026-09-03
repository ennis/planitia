use clap::Parser;
use regex::Regex;
use roxmltree::Node;
use std::borrow::Cow;
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

#[derive(Copy, Clone, Eq, PartialEq)]
enum DispatchType {
    Entry,
    Instance,
    Device,
}
type CommandDispatchTypes = HashMap<String, DispatchType>;
type EnumTypeMap = HashMap<String, String>;

fn generate(out_dir: &Path, document: &roxmltree::Document) -> io::Result<()> {
    let root = document.root();
    let registry = child_tagged(root, "registry").unwrap();

    // map enum_name -> enum type
    // necessary for <enum> entries in vk.xml which have only an alias, with no way to deduce the
    // type
    let mut enum_types = EnumTypeMap::new();
    let mut dispatch_types = CommandDispatchTypes::new();

    // src/vk.rs
    {
        let out_file = File::create(out_dir.join("src/vk.rs"))?;
        let mut buf_writer = BufWriter::new(out_file);
        let mut line_writer = LineWriter::new(IndentedWriter::new(&mut buf_writer));
        gen_preamble(&mut line_writer)?;
        gen_types(&mut line_writer, registry)?;
        gen_enums(&mut line_writer, registry, &mut enum_types)?;
        gen_feature_enums(&mut line_writer, registry, &mut enum_types)?;
        gen_extension_enums(&mut line_writer, registry, &mut enum_types)?;
        gen_commands(&mut line_writer, registry, &mut dispatch_types)?;
        gen_core_dispatch_tables(&mut line_writer, registry, &mut dispatch_types)?;
    }

    Ok(())
}

static PREAMBLE: &str = r#"use crate::macros::*;   // handle, nondispatchable_handle
use std::ffi::{c_void, c_char};
use std::ptr;

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

fn gen_types(out: &mut Writer, registry: Node) -> io::Result<()> {
    for types in children_tagged(registry, "types") {
        for ty in children_tagged(types, "type") {
            gen_type(out, ty)?;
        }
    }
    Ok(())
}

fn gen_type(out: &mut Writer, node: Node) -> io::Result<()> {
    if let Some(api) = node.attribute("api")
        && api == "vulkansc"
    {
        return Ok(());
    }

    let Some(category) = node.attribute("category") else {
        return Ok(());
    };

    // PAIN:
    // For bitmasks, the name is in a child tag
    //      unless it's an alias.
    // For enums, the name is in the "name" attribute.
    // For handles, it's usually in a child tag.
    // For <funcpointer>, it's within a <name> tag inside <proto>
    //
    // Who cares about consistency?
    //
    // PAIN: Also, base types (e.g. `VkBool32`, `VkFlags`, etc.) are not generated there because
    //       generating them from vk.xml is intractable (some of them contain preprocessor directives...)
    let name = sanitize_ident(
        node.attribute("name")
            .or_else(|| child_tagged(node, "name").map(|n| n.text().unwrap()))
            .or_else(|| child_tagged(node, "proto").and_then(|p| child_tagged(p, "name")).map(|n| n.text().unwrap()))
            .unwrap(),
    );
    if let Some(alias) = node.attribute("alias").map(sanitize_ident) {
        write!(out, "pub type {name} = {alias};\n")?;
        return Ok(());
    }
    match category {
        "enum" => {
            // PAIN: Enum <type> records don't actually specify their type, so we can't generate them here.
        }
        "bitmask" => {
            let ty = child_tagged(node, "type").map(|n| n.text().unwrap()).unwrap();
            let rust_ty = c_type_to_rust(ty);
            write!(out, "pub type {name} = {rust_ty};\n")?;
        }
        "handle" => {
            // For handles
            let ty = child_tagged(node, "type").map(|n| n.text().unwrap()).unwrap();
            match ty {
                "VK_DEFINE_HANDLE" => {
                    write!(out, "handle!({name}, {name}_T);\n")?;
                }
                "VK_DEFINE_NON_DISPATCHABLE_HANDLE" => {
                    write!(out, "non_dispatchable_handle!({name});\n")?;
                }
                _ => {
                    panic!("unexpected handle type");
                }
            }
        }
        "struct" => {
            writeln!(out, "/// <https://docs.vulkan.org/refpages/latest/refpages/source/{name}.html>")?;
            writeln!(out, "#[repr(C)]")?;
            writeln!(out, "#[cfg_attr(feature = \"debug\", derive(Debug))]")?;
            writeln!(out, "#[derive(Copy, Clone)]")?;
            write!(out, "pub struct {name} {{\n")?;
            indent(out);
            for member in children_tagged(node, "member") {
                if let Some(api) = member.attribute("api")
                    && api == "vulkansc"
                {
                    continue;
                }

                // PAIN: The text of <member> is actally C syntax for a struct member.
                //       There is markup for the <name> and <type> but the cv-qualifiers, pointer & array declarators
                //       are not marked up, so they are basically useless and we ignore them.
                //       Parse the C syntax directly instead.

                let text = node_text(&member);
                let values = member.attribute("values");
                let optional = member.attribute("optional").unwrap_or("false").split(',').collect::<Vec<_>>();
                let last_opt = optional.last().map(|&s| s == "true").unwrap_or(false);
                let len = member.attribute("len");

                write!(out, "pub ")?;
                let member_decl = convert_c_declarator(out, &text)?;
                // Write default field values.
                match member_decl.name {
                    // sType field
                    "sType" if values.is_some() => write!(out, " = {}", values.unwrap())?,
                    // pNext is always defaultable to null
                    "pNext" => {
                        write!(out, " = {}", if member_decl.const_ptr { "ptr::null()" } else { "ptr::null_mut()" })?
                    }
                    // pointers with len attributes are defaultable to null (if len == 0)
                    _ if len.is_some() && member_decl.ptr => {
                        write!(out, " = {}", if member_decl.const_ptr { "ptr::null()" } else { "ptr::null_mut()" })?
                    }
                    // otherwise, decide based on the "optional" field
                    _other if last_opt => {
                        if member_decl.const_ptr {
                            write!(out, " = ptr::null()")?;
                        } else if member_decl.ptr {
                            write!(out, " = ptr::null_mut()")?;
                        } else {
                            write!(out, " = 0")?;
                        }
                    }
                    _ => {}
                }
                writeln!(out, ",")?;
            }
            dedent(out);
            writeln!(out, "}}")?;
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
        let ty = en.attribute("type").map(c_type_to_rust);
        write!(out, "pub const {}: ", name)?;
        let ty = match ty {
            Some(ty) => c_type_to_rust(ty),
            None => ty_name.as_ref().unwrap().as_ref(),
        };
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

fn gen_command(out: &mut Writer, cmd: Node, cmd_dispatch_types: &mut CommandDispatchTypes) -> io::Result<()> {
    if !api_is_vulkan(cmd) {
        return Ok(());
    }
    if let Some(alias) = cmd.attribute("alias") {
        let name = cmd.attribute("name").unwrap();
        write!(out, "pub type PFN_{name} = PFN_{alias};\n")?;
        cmd_dispatch_types.insert(name.to_string(), *cmd_dispatch_types.get(alias).unwrap());
        return Ok(());
    }
    let proto = child_tagged(cmd, "proto").unwrap();
    let ret_type = c_type_to_rust(child_tagged(proto, "type").unwrap().text().unwrap());
    let name = sanitize_ident(child_tagged(proto, "name").unwrap().text().unwrap());
    write!(out, "pub type PFN_{name} = unsafe extern \"system\" fn(")?;
    let mut first = true;
    let mut first_param = None;
    for param in children_tagged(cmd, "param") {
        if !first {
            write!(out, ", ")?;
        }
        if first {
            first_param = Some(param.clone());
        }
        first = false;
        let decl = node_text(&param);
        convert_c_declarator(out, &decl)?;
    }
    write!(out, ") -> {ret_type};\n")?;
    // Infer command type from first param
    let first_param_ty = first_param.and_then(|param| child_tagged(param, "type")).and_then(|n| n.text());
    cmd_dispatch_types.insert(name.to_string(), dispatch_type_from_first_param(first_param_ty));
    Ok(())
}

fn gen_commands(out: &mut Writer, registry: Node, cmd_type_map: &mut CommandDispatchTypes) -> io::Result<()> {
    let commands = child_tagged(registry, "commands").unwrap();
    for cmd in children_tagged(commands, "command") {
        gen_command(out, cmd, cmd_type_map)?;
    }
    Ok(())
}

fn feature_commands<'a, 'input>(features: &Node<'a, 'input>, feature: &Node<'a, 'input>) -> Vec<Node<'a, 'input>> {
    // PAIN: "depends" is defined by the spec to be a boolean expression with AND, OR and arbitrary groups;
    //       this would make it very difficult to build the set of supported commands
    //       fortunately, vk.xml seems to only use '+' in core features, so we don't have to
    //
    let depnames = feature.attribute("depends").map(|s| s.split('+').collect::<Vec<_>>()).unwrap_or_default();
    let deps = depnames
        .iter()
        .map(|&dep| {
            children_tagged(*features, "feature")
                .find(|f| f.attribute("name").unwrap() == dep)
                .unwrap_or_else(|| panic!("feature {dep} not found"))
        })
        .collect::<Vec<_>>();
    let mut recdeps = deps.iter().map(|dep| feature_commands(features, dep)).flatten().collect::<Vec<_>>();
    recdeps.push(feature.clone());
    recdeps.dedup_by_key(|f| f.attribute("name").unwrap());
    recdeps
}

fn gen_core_dispatch_tables(out: &mut Writer, registry: Node, dispatch_types: &CommandDispatchTypes) -> io::Result<()> {
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
                    match &dispatch_types[cmd.attribute("name").unwrap()] {
                        DispatchType::Entry => entry_fns.push(cmd),
                        DispatchType::Instance => instance_fns.push(cmd),
                        DispatchType::Device => device_fns.push(cmd),
                    }
                }
            }
        }
        writeln!(out, "// Vulkan {api_version}")?;
        if !entry_fns.is_empty() {
            writeln!(out, "#[derive(Copy, Clone)]")?;
            writeln!(out, "pub struct Vulkan_{api_version}_EntryDispatch {{")?;
            for cmd in entry_fns.iter() {
                let vk_cmd_name = cmd.attribute("name").unwrap();
                let cmd_name = vk_cmd_name.strip_prefix("vk").unwrap();
                writeln!(out, "    pub {cmd_name}: PFN_{vk_cmd_name},")?;
            }
            writeln!(out, "}}")?;
        }
        if !instance_fns.is_empty() {
            writeln!(out, "#[derive(Copy, Clone)]")?;
            writeln!(out, "pub struct Vulkan_{api_version}_InstanceDispatch {{")?;
            for cmd in instance_fns.iter() {
                let vk_cmd_name = cmd.attribute("name").unwrap();
                let cmd_name = vk_cmd_name.strip_prefix("vk").unwrap();
                writeln!(out, "    pub {cmd_name}: PFN_{vk_cmd_name},")?;
            }
            writeln!(out, "}}")?;
        }
        if !device_fns.is_empty() {
            writeln!(out, "#[derive(Copy, Clone)]")?;
            writeln!(out, "pub struct Vulkan_{api_version}_DeviceDispatch {{")?;
            for cmd in device_fns.iter() {
                let vk_cmd_name = cmd.attribute("name").unwrap();
                let cmd_name = vk_cmd_name.strip_prefix("vk").unwrap();
                writeln!(out, "    pub {cmd_name}: PFN_{vk_cmd_name},")?;
            }
            writeln!(out, "}}")?;
        }
    }
    Ok(())
}

fn gen_require(
    out: &mut Writer,
    require: Node,
    extnumber: Option<i64>,
    enum_types: &mut EnumTypeMap,
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
    Ok(())
}

fn gen_extension_enums(out: &mut Writer, registry: Node, enum_types: &mut EnumTypeMap) -> io::Result<()> {
    let extensions = child_tagged(registry, "extensions").unwrap();
    for ext in children_tagged(extensions, "extension") {
        if !is_vulkan_supported_extension(ext) {
            continue;
        }
        let name = ext.attribute("name").unwrap();
        let number = int_attr(ext, "number").unwrap();
        writeln!(out, "// Extension: {name} ({number})")?;
        for require in children_tagged(ext, "require") {
            gen_require(out, require, Some(number), enum_types)?;
        }
    }
    Ok(())
}

fn gen_feature_enums(out: &mut Writer, registry: Node, enum_types: &mut EnumTypeMap) -> io::Result<()> {
    for feature in children_tagged(registry, "feature") {
        let name = feature.attribute("name").unwrap();
        //let number = int_attr(feature, "number").unwrap();
        writeln!(out, "// Feature: {name}")?;
        for require in children_tagged(feature, "require") {
            gen_require(out, require, None, enum_types)?;
        }
    }
    Ok(())
}

//--------------------------------------------------------------------------------------------------
// Helpers
//--------------------------------------------------------------------------------------------------

fn api_is_vulkan(node: Node) -> bool {
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
struct CDecl<'a> {
    /// Name (e.g. `name` in `const char* name[10]`)
    name: &'a str,
    /// True if this is a const-qualified pointer.
    const_ptr: bool,
    /// True if this is a pointer.
    ptr: bool,
}

/// Converts a C declarator to its Rust equivalent and writes it to `out`.
///
/// Returns parsed information about the declarator.
///
/// # Example
/// `const char* name[10]` -> `name: [*const c_char; 10]`, returns (`name`, false)
/// `struct VkBaseInStructure* pNext` -> `pNext: *mut VkBaseInStructure`, returns (`pNext`, false)
fn convert_c_declarator<'a>(out: &mut Writer, member: &'a str) -> io::Result<CDecl<'a>> {
    // [const] [struct] <type> [*|const *] <name> [ [<array_len>] ] [: bitfield]
    // https://regex101.com/?regex=%5E%5Cs*%28const%5Cs%2B%29%3F%28struct%5Cs%2B%29%3F%28%5Cw%2B%29%5Cs*%28%5C*%29%3F%5Cs*%28const%29%3F%5Cs*%28%5C*%29%3F%5Cs*%28%5Cw%2B%29%5Cs*%28%3F%3A%5C%5B%5Cs*%28%5Cw%2B%29%5Cs*%5C%5D%29%3F%24&testString=const++char32*+const*+ppEnabledExtensionNames+%5B4%5D%0Aconst+++struct+++VkBaseInStructure*+pNext%0AVkStructureType+sType%0Auint8_t+pipelineCacheUUID%5BVK_UUID_SIZE%5D%0Avoid**+ppData%0A&flags=gmu&flavor=rust&delimiter=%22
    static DECL_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^\s*(const\s+)?(struct\s+)?(\w+)\s*(\*)?\s*(const)?\s*(\*)?\s*(\w+)\s*(?:\[\s*(\w+)\s*])?\s*(?:\[\s*(\w+)\s*])?\s*(?::\s*(\d+))?\s*$")
            .unwrap()
    });
    // 1 inner const
    // 2 struct
    // 3 type
    // 4 inner ptr
    // 5 outer const
    // 6 outer ptr
    // 7 name
    // 8 array len
    // 9 bitfield
    let caps = DECL_RE.captures(member).unwrap();
    let inner_const = caps.get(1).map(|m| m.as_str()).is_some();
    //let struct_ = caps.get(2).map(|m| m.as_str());
    let ty = caps.get(3).map(|m| m.as_str()).unwrap();
    let inner_ptr = caps.get(4).map(|m| m.as_str()).is_some();
    let outer_const = caps.get(5).map(|m| m.as_str()).is_some();
    let outer_ptr = caps.get(6).map(|m| m.as_str()).is_some();
    let name = caps.get(7).map(|m| m.as_str()).unwrap();
    let array_len_outer = caps.get(8).map(|m| m.as_str());
    let array_len_inner = caps.get(9).map(|m| m.as_str());
    let bitfield = caps.get(10).map(|m| m.as_str());
    let mut ty = c_type_to_rust(ty).to_string();
    if inner_ptr {
        if inner_const {
            ty = format!("*const {ty}");
        } else {
            ty = format!("*mut {ty}");
        }
    }
    if outer_ptr {
        if outer_const {
            ty = format!("*const {ty}");
        } else {
            ty = format!("*mut {ty}");
        }
    }
    if let Some(array_len) = array_len_inner {
        ty = format!("[{ty}; {} as usize]", array_len);
    }
    if let Some(array_len) = array_len_outer {
        ty = format!("[{ty}; {} as usize]", array_len);
    }
    let sane_name = sanitize_ident(name);
    write!(out, "{sane_name}: {ty}")?;
    // Quoting vk.xml:
    //
    //  > "The bitfields in this structure are non-normative since bitfield ordering is
    //     implementation-defined in C. The specification defines the normative layout."
    //
    // So vk.xml is not a *complete* definition of the API, you still have to write some stuff
    // by hand. Great.
    if let Some(bitfield) = bitfield {
        write!(out, " /* bitfield: {bitfield} */")?;
    }
    let ptr = outer_ptr || inner_ptr;
    let const_ = (outer_ptr && outer_const) || (!outer_const && inner_ptr && inner_const);
    Ok(CDecl { name, const_ptr: const_, ptr })
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
fn c_type_to_rust(ty: &str) -> &str {
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
        "void" => "c_void",
        "char" => "c_char",
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
