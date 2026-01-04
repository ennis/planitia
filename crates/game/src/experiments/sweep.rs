use gamelib::asset::Handle;
use geom::GeoArchive;
use gpu::Buffer;

pub struct SweepExperiment {
    sweep_stroke_pipeline: Handle<gpu::ComputePipeline>,
    debug_stroke_pipeline: Handle<gpu::GraphicsPipeline>,

    strokes: Buffer<[geom::coat::SweptStroke]>,
    stroke_vertices: Buffer<[geom::coat::SweptStrokeVertex]>,
    cross_sections: Buffer<[geom::coat::PosNorm2DVertex]>,
}

impl SweepExperiment {
    pub fn new() -> Self {
        let sweep_stroke_pipeline =
            gamelib::pipeline_cache::get_compute_pipeline("/shaders/pipelines.parc#sweep_strokes");
        let debug_stroke_pipeline =
            gamelib::pipeline_cache::get_graphics_pipeline("/shaders/pipelines.parc#debug_swept_strokes");

        let strokes = Buffer::from_slice(&[], "swept_strokes");
        let stroke_vertices = Buffer::from_slice(&[], "swept_stroke_vertices");
        let cross_sections = Buffer::from_slice(&[], "swept_stroke_cross_sections");

        Self {
            sweep_stroke_pipeline,
            debug_stroke_pipeline,
            strokes,
            stroke_vertices,
            cross_sections,
        }
    }
}
