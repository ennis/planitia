use crate::cfg;
use anyhow::Result;
use color::Srgba8;
use color_print::{ceprintln, cprintln};
use geom::coat::Coat;
use geom::mesh::{Mesh, MeshPart};
use geom::{GeoArchiveHeader, SweptStroke, SweptStrokeVertex};
use hgeo::{AttributeClass, Geo, Group, StorageKind};
use math::{vec2, Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::path::Path;
use utils::archive::{ArchiveWriter, Offset};

pub struct ConvertConfig {
    /// Output geometry file.
    pub output_file: Option<String>,
}

pub fn convert_geo_file(geo_file: impl AsRef<Path>, config: &ConvertConfig) {
    let output_file = get_output_file_name(geo_file.as_ref(), config).unwrap();
    let mut state = Converter::new();
    state.convert(geo_file.as_ref(), config).unwrap();
    state.finish(Path::new(&output_file)).unwrap()
}

fn get_output_file_name(geo_file: &Path, config: &ConvertConfig) -> Result<String> {
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

#[derive(Clone,Default)]
struct CrossSection {
    name: String,
    start_vertex: u32,
    vertex_count: u32,
}

struct Converter {
    archive: ArchiveWriter,
    header: Offset<GeoArchiveHeader>,
    /// Generic surface mesh vertices.
    mesh_vertices: Vec<geom::mesh::MeshVertex>,
    /// Paint stroke vertices
    stroke_vertices: Vec<geom::coat::StrokeVertex>,
    /// Swept stroke vertices
    swept_stroke_vertices: Vec<geom::coat::SweptStrokeVertex>,
    /// 2D position+normal vertices
    cross_section_vertices: Vec<geom::coat::PosNorm2DVertex>,
    mesh_indices: Vec<u32>,
    /// Paint stroke primitives
    strokes: Vec<geom::coat::Stroke>,
    /// Swept stroke primitives
    swept_strokes: Vec<geom::coat::SweptStroke>,
    /// Coat primitives
    coats: Vec<geom::coat::Coat>,
    /// Surface meshes
    meshes: Vec<geom::mesh::Mesh>,
    cross_sections: Vec<CrossSection>,
}

impl Converter {
    fn new() -> Converter {
        let mut archive = ArchiveWriter::new();
        let header = archive.write(geom::GeoArchiveHeader {
            magic: geom::GeoArchiveHeader::MAGIC,
            version: geom::GeoArchiveHeader::VERSION,
            ..
        });

        Converter {
            archive,
            header,
            mesh_vertices: vec![],
            stroke_vertices: vec![],
            swept_stroke_vertices: vec![],
            cross_section_vertices: vec![],
            mesh_indices: vec![],
            strokes: vec![],
            swept_strokes: vec![],
            coats: vec![],
            meshes: vec![],
            cross_sections: vec![],
        }
    }

    fn finish(self, output_file: &Path) -> Result<()> {
        let mut archive = self.archive;
        let header = self.header;

        let coats = archive.write_slice(&self.coats);
        let mesh_vertices = archive.write_slice(&self.mesh_vertices);
        let mesh_indices = archive.write_slice(&self.mesh_indices);
        let stroke_vertices = archive.write_slice(&self.stroke_vertices);
        let strokes = archive.write_slice(&self.strokes);
        let swept_strokes = archive.write_slice(&self.swept_strokes);
        let meshes = archive.write_slice(&self.meshes);
        let swept_stroke_vertices = archive.write_slice(&self.swept_stroke_vertices);
        let cross_section_vertices = archive.write_slice(&self.cross_section_vertices);

        let arrays = archive.write_slice(&[
            geom::VertexArray::Stroke(stroke_vertices),
            geom::VertexArray::Mesh(mesh_vertices),
            geom::VertexArray::SweptStroke(swept_stroke_vertices),
            geom::VertexArray::PosNorm2D(cross_section_vertices),
        ]);

        archive[header].indices = mesh_indices;
        archive[header].vertex_arrays = arrays;
        archive[header].primitives = archive.write_slice(&[
            geom::Primitive::Stroke(strokes),
            geom::Primitive::Coat(coats),
            geom::Primitive::Mesh(meshes),
            geom::Primitive::SweptStroke(swept_strokes),
        ]);

        if !cfg().quiet {
            cprintln!(
                "<bold>Writing</bold> geometry ({} coat(s), {} strokes, {} cross sections, {} mesh vertices, {} mesh indices, {} stroke vertices)",
                self.coats.len(),
                self.strokes.len(),
                self.cross_sections.len(),
                self.mesh_vertices.len(),
                self.mesh_indices.len(),
                self.stroke_vertices.len()
            );
        }

        // write output file
        archive.write_to_file(&output_file)?;

        Ok(())
    }

    fn convert(&mut self, geo_file: &Path, config: &ConvertConfig) -> Result<()> {
        let geo = Geo::load(geo_file)?;

        self.convert_coats(&geo)?;
        self.convert_mesh(&geo)?;
        self.convert_cross_sections(&geo)?;

        Ok(())
    }

    fn required_attribs(&mut self, g: &Geo, required_attribs: &[(&str, AttributeClass, StorageKind, usize)]) {
        let quiet = cfg().quiet;
        if !quiet {
            for (name, attr_class, kind, size) in required_attribs {
                let Some(attribute) = g.attribute(*attr_class, name) else {
                    ceprintln!("<y,bold>warning:</> missing point attribute `{}`", name);
                    continue;
                };
                if attribute.storage_kind() != *kind || attribute.size != *size {
                    ceprintln!("<y,bold>warning:</> point attribute `{}` has invalid type", name);
                }
            }
        }
    }

    /// Converts 3D polygon data.
    fn convert_mesh(&mut self, g: &Geo) -> anyhow::Result<()> {
        // whether the point index has been added to the vertex list
        // (houdini point index -> our mesh vertex index)
        let mut inserted_points = HashMap::<u32, u32>::new();

        let mut cur_prim_indices = Vec::new();
        let start_index = self.mesh_indices.len() as u32;
        let base_vertex = self.mesh_vertices.len() as u32;
        let mut index_count = 0u32;

        for prim in g.polygons() {
            if !prim.closed {
                // skip polylines
                continue;
            }

            for (i, vi) in prim.vertices().enumerate() {
                let pi = g.vertexpoint(vi);

                // emit vertex
                let v = *inserted_points.entry(pi).or_insert_with(|| {
                    let position: Vec3 = g.point(pi, "P");
                    let normal: Vec3 = g.vertex(vi, "N");
                    let color: Vec3 = g.point(pi, "Cd");
                    let uv: Vec2 = g.vertex(vi, "uv");

                    self.mesh_vertices.push(geom::mesh::MeshVertex {
                        position,
                        normal,
                        color: Srgba8::from_linear(color.x, color.y, color.z, 1.0),
                        uv,
                    });

                    let vertex_index = self.mesh_vertices.len() as u32 - base_vertex;
                    vertex_index
                });

                cur_prim_indices.push(v);

                if i >= 2 {
                    // emit triangle
                    self.mesh_indices.push(cur_prim_indices[0]);
                    self.mesh_indices.push(cur_prim_indices[i - 1]);
                    self.mesh_indices.push(cur_prim_indices[i]);
                    index_count += 3;
                }
            }

            cur_prim_indices.clear()
        }

        self.meshes.push(Mesh {
            parts: self.archive.write_slice(&[MeshPart {
                start_index,
                index_count,
                base_vertex,
            }]),
        });
        Ok(())
    }

    fn convert_coats(&mut self, g: &Geo) -> anyhow::Result<()> {
        // coats are groups of stroke primitives that start with "coat".
        for (group_name, group) in g.primitive_groups() {
            if group_name.starts_with("coat") {
                self.convert_coat(g, group)?;
            }
        }
        Ok(())
    }

    fn convert_coat(&mut self, g: &Geo, coat_group: &Group) -> anyhow::Result<()> {
        self.required_attribs(
            g,
            &[
                ("N", AttributeClass::Point, StorageKind::FpReal32, 3),
                ("arclen", AttributeClass::Point, StorageKind::FpReal32, 1),
                ("Cd", AttributeClass::Point, StorageKind::FpReal32, 3),
                ("width", AttributeClass::Point, StorageKind::FpReal32, 3),
                ("tangent", AttributeClass::Point, StorageKind::FpReal32, 3),
            ],
        );

        let start_vertex = self.stroke_vertices.len() as u32;
        let start_stroke = self.strokes.len() as u32;

        for polyline in g.polylines_in_group(coat_group) {
            let stroke_start_vertex = self.stroke_vertices.len() as u32;

            for vertex in polyline.vertices() {
                let point = g.vertexpoint(vertex);
                let position: Vec3 = g.vertex(vertex, "P");
                let normal: Vec3 = g.vertex(vertex, "N");
                let tangent: Vec3 = g.vertex(vertex, "tangentu");
                let arclength: f32 = g.vertex(vertex, "arclen");
                let color: Vec3 = g.point(point, "Cd");
                //let width: f32 = geo.vertexattrib(vertex, "width");

                self.stroke_vertices.push(geom::coat::StrokeVertex {
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

            self.strokes.push(geom::coat::Stroke {
                ref_position: Default::default(), // TODO export reference position on surface mesh
                start_vertex: stroke_start_vertex,
                vertex_count: polyline.vertex_count as u32,
                brush_index: 0,   // TODO
                width_profile: 0, // TODO
                color_ramp: 0,    // TODO
            });
        }

        self.coats.push(Coat {
            start_stroke,
            stroke_count: self.strokes.len() as u32 - start_stroke,
            start_vertex,
            vertex_count: self.stroke_vertices.len() as u32 - start_vertex,
            color_ramps: Offset::INVALID,    // TODO
            width_profiles: Offset::INVALID, // TODO
        });
        Ok(())
    }

    fn convert_swept_strokes(&mut self, g: &Geo) -> Result<()> {
        let group = g.primitive_group("backbones");
        let cross_section = self.cross_sections.first().cloned().unwrap_or_default();

        for polyline in g.polylines_in_group(group) {
            let start_vertex = self.swept_stroke_vertices.len() as u32;
            let vertex_count = polyline.vertex_count as u32;
            for vertex in polyline.vertices() {
                let position = g.vertex(vertex, "P");
                let normal = g.vertex(vertex, "N");
                let up = g.vertex(vertex, "up");
                let color: Vec4 = g.vertex(vertex, "Cd");

                self.swept_stroke_vertices.push(SweptStrokeVertex {
                    position,
                    color: Srgba8::from_linear(color.x, color.y, color.z, 1.0),
                    normal,
                    up,
                })
            }
            self.swept_strokes.push(SweptStroke {
                ref_position: Default::default(),
                start_vertex,
                vertex_count,
                cross_section_start_vertex: cross_section.start_vertex,
                cross_section_vertex_count: cross_section.vertex_count,
            })
        }
        Ok(())
    }

    fn convert_cross_sections(&mut self, g: &Geo) -> Result<()> {
        for (group_name, group) in g.primitive_groups() {
            if group_name.starts_with("cross_section") {
                self.convert_cross_section(g, group, group_name)?
            }
        }
        Ok(())
    }

    ///
    fn convert_cross_section(&mut self, g: &Geo, group: &Group, name: &str) -> Result<()> {
        self.required_attribs(g, &[("N", AttributeClass::Point, StorageKind::FpReal32, 3)]);

        let start_vertex = self.stroke_vertices.len() as u32;
        let polyline = g
            .polylines_in_group(group)
            .next()
            .ok_or_else(|| anyhow::anyhow!("no polyline found in group"))?;
        let vertex_count = polyline.vertex_count as u32;
        for vertex in polyline.vertices() {
            let point = g.vertexpoint(vertex);
            let position: Vec3 = g.vertex(vertex, "P");
            let normal: Vec3 = g.vertex(vertex, "N");
            self.cross_section_vertices.push(geom::coat::PosNorm2DVertex {
                position: vec2(position.x, position.y),
                normal: vec2(normal.x, normal.y),
            });
        }

        self.cross_sections.push(CrossSection {
            name: name.to_string(),
            start_vertex,
            vertex_count,
        });

        Ok(())
    }
}
