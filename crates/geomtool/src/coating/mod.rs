use crate::houdini::{Geo, StorageKind};
use crate::{Config, cfg, houdini};
use color::Srgba8;
use color_print::{ceprintln, cprintln};
use geom::coat::Coat;
use geom::mesh::{Mesh, MeshPart};
use math::{Vec2, Vec3, Vec4};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use utils::archive::{ArchiveWriter, Offset};

pub struct CoatingConfig {
    /// Output geometry file.
    pub output_file: Option<String>,
}

pub fn create_coating_mesh(geo_file: impl AsRef<Path>, config: &CoatingConfig) {
    create_coating_mesh_inner(geo_file.as_ref(), config).unwrap();
}

fn get_output_file_name(geo_file: &Path, config: &CoatingConfig) -> anyhow::Result<String> {
    match &config.output_file {
        Some(f) => Ok(f.clone()),
        None => {
            let stem = geo_file
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow::anyhow!("invalid geo file name"))?;
            Ok(format!("{}.geoarc", stem))
        }
    }
}

#[derive(Default)]
struct GeometryBuffers {
    mesh_vertices: Vec<geom::mesh::MeshVertex>,
    stroke_vertices: Vec<geom::coat::StrokeVertex>,
    cross_section_vertices: Vec<geom::coat::CrossSectionVertex>,
    mesh_indices: Vec<u32>,
    strokes: Vec<geom::coat::Stroke>,
}

fn create_coating_mesh_inner(geo_file: &Path, config: &CoatingConfig) -> anyhow::Result<()> {
    let quiet = cfg().quiet;

    let output_file = get_output_file_name(geo_file, config)?;

    let mut archive = ArchiveWriter::new();

    let header = archive.write(geom::GeoArchiveHeader {
        magic: geom::GeoArchiveHeader::MAGIC,
        version: geom::GeoArchiveHeader::VERSION,
        ..
    });

    // import bgeo/.geo file
    let geo = houdini::Geo::load(geo_file)?;
    let mut buffers = GeometryBuffers::default();

    // convert coat groups
    let mut coats = Vec::new();
    for (group_name, group) in geo.primitive_groups() {
        if group_name.starts_with("coat") {
            coats.push(convert_coat(&mut archive, &geo, group, &mut buffers)?);
        }
    }

    // convert base mesh
    let mesh = convert_mesh(&mut archive, &geo, &mut buffers)?;

    if !quiet {
        cprintln!(
            "<bold>Writing</bold> geometry to `{}` ({} coat(s), {} strokes, {} mesh vertices, {} mesh indices, {} stroke vertices)",
            output_file,
            coats.len(),
            buffers.strokes.len(),
            buffers.mesh_vertices.len(),
            buffers.mesh_indices.len(),
            buffers.stroke_vertices.len()
        );
    }

    let coats = archive.write_slice(&coats);
    let mesh_vertices = archive.write_slice(&buffers.mesh_vertices);
    let mesh_indices = archive.write_slice(&buffers.mesh_indices);
    let stroke_vertices = archive.write_slice(&buffers.stroke_vertices);
    let strokes = archive.write_slice(&buffers.strokes);
    let meshes = archive.write_slice(&[mesh]);

    archive[header].stroke_vertices = stroke_vertices;
    archive[header].mesh_vertices = mesh_vertices;
    archive[header].mesh_indices = mesh_indices;
    archive[header].strokes = strokes;
    archive[header].coats = coats;
    archive[header].meshes = meshes;

    // write output file
    archive.write_to_file(&output_file)?;

    Ok(())
}

fn convert_mesh(
    archive: &mut ArchiveWriter,
    geo: &houdini::Geo,
    buffers: &mut GeometryBuffers,
) -> anyhow::Result<Mesh> {
    // whether the point index has been added to the vertex list
    // (houdini point index -> our mesh vertex index)
    let mut inserted_points = HashMap::<u32, u32>::new();

    let mut cur_prim_indices = Vec::new();
    let start_index = buffers.mesh_indices.len() as u32;
    let base_vertex = buffers.mesh_vertices.len() as u32;
    let mut index_count = 0u32;

    for prim in geo.iter_polygons() {
        if !prim.closed {
            // skip polylines
            continue;
        }

        for (i, vi) in prim.vertices().enumerate() {
            let pi = geo.vertexpoint(vi);

            // emit vertex
            let v = *inserted_points.entry(pi).or_insert_with(|| {
                let position: Vec3 = geo.pointattrib(pi, "P");
                let normal: Vec3 = geo.vertexattrib(vi, "N");
                let color: Vec3 = geo.pointattrib(pi, "Cd");
                let uv: Vec2 = geo.vertexattrib(vi, "uv");

                buffers.mesh_vertices.push(geom::mesh::MeshVertex {
                    position,
                    normal,
                    color: Srgba8::from_linear(color.x, color.y, color.z, 1.0),
                    uv,
                });

                let vertex_index = buffers.mesh_vertices.len() as u32 - base_vertex;
                vertex_index
            });

            cur_prim_indices.push(v);

            if i >= 2 {
                // emit triangle
                buffers.mesh_indices.push(cur_prim_indices[0]);
                buffers.mesh_indices.push(cur_prim_indices[i - 1]);
                buffers.mesh_indices.push(cur_prim_indices[i]);
                index_count += 3;
            }
        }

        cur_prim_indices.clear()
    }

    Ok(Mesh {
        parts: archive.write_slice(&[geom::mesh::MeshPart {
            start_index,
            index_count,
            base_vertex,
        }]),
    })
}

fn warn_required_attributes(geo: &houdini::Geo, required_point_attribs: &[(&str, StorageKind, usize)]) {
    let quiet = cfg().quiet;
    if !quiet {
        for attr in required_point_attribs {
            let Some(attribute) = geo.point_attribute(attr.0) else {
                ceprintln!("<y,bold>warning:</> missing point attribute `{}`", attr.0);
                continue;
            };
            if attribute.storage_kind() != attr.1 || attribute.size != attr.2 {
                ceprintln!("<y,bold>warning:</> point attribute `{}` has invalid type", attr.0);
            }
        }
    }
}

fn convert_coat(
    archive: &mut ArchiveWriter,
    geo: &houdini::Geo,
    coat_group: &houdini::Group,
    buffers: &mut GeometryBuffers,
) -> anyhow::Result<Coat> {
    warn_required_attributes(
        geo,
        &[
            ("N", StorageKind::FpReal32, 3),
            ("arclen", StorageKind::FpReal32, 1),
            ("Cd", StorageKind::FpReal32, 3),
            ("width", StorageKind::FpReal32, 3),
            ("tangent", StorageKind::FpReal32, 3),
        ],
    );

    let start_vertex = buffers.stroke_vertices.len() as u32;
    let start_stroke = buffers.strokes.len() as u32;

    for polyline in geo.iter_polylines().filter(|p| coat_group.contains(p.primitive_index)) {
        let stroke_start_vertex = buffers.stroke_vertices.len() as u32;

        for vertex in polyline.vertices() {
            let point = geo.vertexpoint(vertex);
            let position: Vec3 = geo.vertexattrib(vertex, "P");
            let normal: Vec3 = geo.vertexattrib(vertex, "N");
            let tangent: Vec3 = geo.vertexattrib(vertex, "tangentu");
            let arclength: f32 = geo.vertexattrib(vertex, "arclen");
            let color: Vec3 = geo.pointattrib(point, "Cd");
            //let width: f32 = geo.vertexattrib(vertex, "width");

            buffers.stroke_vertices.push(geom::coat::StrokeVertex {
                position,
                color: Srgba8::from_linear(color.x, color.y, color.z, 1.0),
                normal,
                arclength,
                tangent,
                /*width: (width * 255.0) as u8, // TODO: what is the correct mapping here? it depends on the scale of the scene
                noise: 0,
                falloff: 0,
                stamp_texture: 0,*/
            });
        }

        buffers.strokes.push(geom::coat::Stroke {
            ref_position: Default::default(), // TODO export reference position on surface mesh
            start_vertex: stroke_start_vertex,
            vertex_count: polyline.vertex_count as u32,
            brush_index: 0,   // TODO
            width_profile: 0, // TODO
            color_ramp: 0,    // TODO
        });
    }

    Ok(Coat {
        start_stroke,
        stroke_count: buffers.strokes.len() as u32 - start_stroke,
        start_vertex,
        vertex_count: buffers.stroke_vertices.len() as u32 - start_vertex,
        color_ramps: Offset::INVALID,    // TODO
        width_profiles: Offset::INVALID, // TODO
    })
}

fn convert_cross_section(
    geo: &houdini::Geo,
    cross_section_group: &houdini::Group,
    buffers: &mut GeometryBuffers,
) -> anyhow::Result<(u32, u32)> {
    warn_required_attributes(geo, &[("N", StorageKind::FpReal32, 3)]);

    let start_vertex = buffers.stroke_vertices.len() as u32;
    let polyline = geo
        .iter_polylines()
        .find(|p| cross_section_group.contains(p.primitive_index))
        .ok_or_else(|| anyhow::anyhow!("no polyline found in group"))?;
    let vertex_count = polyline.vertex_count as u32;
    for vertex in polyline.vertices() {
        let point = geo.vertexpoint(vertex);
        let position: Vec3 = geo.vertexattrib(vertex, "P");
        let normal: Vec3 = geo.vertexattrib(vertex, "N");
        buffers
            .cross_section_vertices
            .push(geom::coat::CrossSectionVertex { position, normal });
    }

    Ok((start_vertex, vertex_count))
}
