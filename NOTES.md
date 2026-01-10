# Goals

- Complex stroke styles, like photoshop or procreate.
- perfect AA (compute accurate subpixel coverage for strokes)
- varying stroke attributes
- textured strokes, with motion-coherent textures

# Next steps

Try pixel linked-list OIT. Draw strokes as splats in compute?

## Better coarse rasterization step

With conservative rasterization the curves appear more than once in each bucket (since it's per-triangle).
Plus the stroking width is a best guess.

## Occlusion culling

Difficult? Possibly based on an existing depth buffer. Or maybe big opaque strokes that are the equivalent of blocking.

## Assigning depth to stroke fragments

Derive fragment depth from depth of nearest curve point? There are options

## Simplification when too many transparent strokes overlap

???
Depends on the final rendering shader

# Textured strokes

## Foreshortening

TODO

## Final blending

The main advantage over HW rasterization is that it's more flexible. However, need to sort fragments by depth, low
occupancy.

Idea: per-tile, allocate just the right amount of memory for sorting ( proportional to the number of strokes )

Will need to look into OIT techniques anyway.
Shouldn't be splat-based, should eval lines/curves per pixel.

**Per-pixel list of: depth + curve index (depth can be omitted).**

## Alternate pipeline

1. Process every curve in parallel, "splat" directly the distance to curve & t value in the fragment buffer.
2. Sort & blend fragments

If "stroke interactions" aren't necessary:

1. Splat the integrated stroke directly in the fragment buffer, cull if alpha = 0 or fully occluded
2. Sort & blend

=> No HW rasterization, but the splatting shader is basically a custom rasterizer.

Alternative: stochastic transparency, removes the need for sorting, but no "stroke interactions"

## Stroke interactions / DF blending

In a nutshell: consider each stroke to be a DF, and allow the DF to be "distorted" by neighboring strokes.
For this, need to defer actual rendering of strokes as late as possible, until we know all stroke DFs interacting with a
pixel.

Basically, it's _non-local_ blending. Typical blending is done per-pixel, this is done at the stroke DF level.

**Unsure how "stroke interactions" are useful in practice**

## Shadows

lol

## Curve shapes

Taper, width profile.

## Ray tracing

# Questions

- shading
- stroke attributes
- stroke SDFs with subpixel accuracy

https://interplayoflight.wordpress.com/2022/06/25/order-independent-transparency-part-1/
https://www.reddit.com/r/GraphicsProgramming/comments/15l8bm9/order_independent_and_indiscriminate_transparency/

# Next steps

## OIT is costly

Fragment interlock tanks perfs. Many overlapping strokes also tanks perfs.
=> Fragment shader of initial rasterization must be ultra cheap!

- Additive blending? May be limited

## Try to repro specific effects

# UI/workflow woes

Too much boilerplate when adding new things:

- adding a pass requires modifying too many places (shaders struct, app ctor, reload_shaders, create_pipeline, render,
  plus much more if it requires resources)
- same with adding a parameter (app struct, ui(), push constants, shaders)
- worst: adding a selectable list of elements by name
    - e.g. brush textures, render mode

700 px
divide in 88 blocks of 8px

render at 88px, 1px = 8px

## Improvements

- creating pipelines: copy/pasting functions, update reload_pipelines, add field in App. Should be easier (PIPELINES)
    - remove the check for null pipeline options
- keeping struct and constants in sync between GLSL & Rust, and also shader interfaces (attachments, arguments) (
  INTERFACES***)
- resizing render targets as the window is resized (RESIZE)
    - to add a new render target, must modify three locations
- allocating and managing temporary render targets (TEMP)
- setting the viewport and scissors & related state (RENDERSTATE)
- allocating render targets with the correct usage (USAGE)
- to add a new UI option, need to change 3 locations (struct field, struct ctor, UI function) (UI)
- lists of options are cumbersome to implement in the UI (UI-LISTS)
- making sure that the format of images matches the shader interface; hard to experiment with because of the need to
  update multiple locations (FORMATS)
- samplers should really be defined next to where they are used, i.e. in shaders (SAMPLERS)
- more generally: adding stuff is just a lot of copy-paste, making the code unreadable; difficult to abstract because
  unclear about requirements of future algorithms
    - a wrong abstraction costs time if in the future it prevents an algorithm from being implemented efficiently
- reuse vertex or mesh+task shaders (REUSE)
- managing one-off image view objects is tedious (IMAGE-VIEWS)

General ideas: more hot-reloading, pipeline as data, GUIs, templates, and sane defaults

Sane defaults:

- viewport & scissors should take the size of the input texture by default

Templates:

- Build passes from templates

## Idea: UI for loading/saving global defines

* Add/remove/enable/disable global defines in the UI.
* On change, recompile all shaders.
* This is just `#define XXX`, no need to pass things in push constants.
* Good for quick tests.

-------------------------------------------------------

# Kinds of painting elements

- **Discrete elements**: leaves, blades of grass, individual strands of hair => something that "exists" (represents a
  concrete object) and is anchored in the world, not view-dependent
    - Goal: flexibility, lighting, shadowing
- **Shading elements**: lighting effects on hair, hair depiction, shadows on cloth, "fibrous" material appearance (like
  wood "figures") => material depiction
    - Goal: reproduce the appearance of overlapping semitransparent strokes
    - at first glance, hair depiction might seem like a "discrete element" problem, but strands of hair are rarely
      depicted individually. The strokes just give the "idea" of hair appearance.
        - it's not _always_ like that, hair depiction really blurs the line between discrete elements and materials
- **Contours**
- **Motion effects**

# Going forward

Taking things seriously:

- a separate application for painting might be too alienating, if the goal is for people to use it; safer to implement
  it as a blender plugin
    - render grease pencil primitives, but augmented with additional attributes, and animate them
    - the core of the application would still be a separate library, sharing its buffers with blender's opengl textures
    - see https://github.com/JiayinCao/SORT/ for an example for custom renderengine
    - also https://docs.blender.org/api/current/bpy.types.RenderEngine.html
- point of comparison: https://gakutada.gumroad.com/l/DeepPaint
- primary goal: get artists (not necessarily from studios) to use it and share their paintings on Twitter (or
  somewhere else)
    - **need to export animated results easily**
    - some people don't know how to animate => need **automatic animation** (turntable, move lights, etc.)
    - wow stuff: a painting that reacts to light changes, viewpoint changes
    - like live2d but "more"
- think about potential clients
- write project summary for submitting to incubators?
- ultimate goal: someone makes a music video with it

# Stroke engine

For actually rendering strokes. Two approaches:

- binned rasterization
- OIT / weighted OIT

Stroke ordering: keeping draw order is important

3D binning: bin curves in 2D + one "depth" or "ordering" dimension

## Idea: Coats

* Coats: group of strokes that have some unity in the painting process
* One render pass per coat / different coats are rendered in different passes.
* Simplified (weighted OIT) blending within a coat
* More complex blending possible between coats

Not all strokes have the same "footprint". Big vs fine details (of course, fine becomes big when zooming in).
How to evaluate the footprint? Depends on stroke width, curvature, curve length.

## Working around high curve counts per tile: depth coats

Assumption: high curve count per tile happens mostly because of camera viewpoints at grazing angles.
In this case: bin curves by screen-space depth. Process depth bins back-to-front.
Selection is done in task/mesh shader (don't split curves between depth bins).
Also, don't split user-defined coats.

1. (Task shader) coat LOD selection from object depth
2. (Mesh shader) emit geometry for curves, assign coat index
3. (Fragment shader) Binning: we have depth, coat index, position. Don't want to split same coat into different depth
   bins?

## Stroke engine parameters

* width procedural
* opacity procedural
* falloff (transverse opacity profile)
* stamp
* color procedural
* blending

## Degenerate strokes

Strokes that point toward or away from the camera. Stroke centerline mostly aligned with view direction.
Very small footprint on the screen because it's facing the camera.

In this case: remove the stroke.

### Golden rule

Strokes that don't face the camera are useless. A meaningful stroke is a stroke that covers the most screen-space area
in relation to its 3D length.

In general: strokes make sense **if they have a significant curve-like footprint on the screen**. I.e. they have to
actually be
strokes, not points.

Observation: most strokes can be embedded into a 3D plane. Consider the normal of this 3D plane. If it's perpendicular
to the screen, then don't draw it (it's a degenerate stroke).
Issue: a lot of strokes are straight lines and are not embedded into only one plane.

## Mixed-order compositing

https://people.csail.mit.edu/ibaran/papers/2011-ASIA-MixedOrder.pdf

Paintings have max 30k strokes; let's target a round 100k strokes per frame. And say 32 subdivs per stroke, that's
3.2 million lines to store in the tiles (possibly more than once, given that lines may affect more than one tile).

At 1920x1080 with 16x16 tiles we have 8100 tiles; thus ~400 lines per tile assuming uniform stroke distribution.

## Transparency

Order:

- for "coats" on a level-set: try draw order. Should be valid outside grazing angles, but then the coat should fade away
  at those anyway.
    - fur, shading elements, "artifact" strokes
- otherwise, for discrete elements, use depth order

Blend modes:

- normal (alpha blending)
- screen
- overlay

Depth order: per-tile VS per-pixel
Generally, per-tile depth sorting isn't correct, high risk of visible discontinuities at tile boundaries

Within a tile, many lines will belong to the same curve. Pixels generated by lines belonging to the same curve will
overlap, and shouldn't blend together.

## Process lines directly

Don't use Bézier curves, use polylines directly.

Bézier curves drawbacks:

- heuristic to calculate the subdiv level is dubious
- hand-drawn strokes don't map well to cubic Bézier curves
- projected curve is a rational Bézier, for which it's complicated to find the tangent/normal

Advantages:

- compact representation
- they can be animated, maybe more easily than polylines (animate the control points)

Polylines should be more flexible, trivial to keyframe, and trivial to expand to triangle strips in a vertex shader.
Possible cull / LOD select in mesh shader.
Assign attribute values to individual points.

# Blending / transparency

The core of the issue. Blend in stroke order for coats, each coat in a separate buffer, then depth-composite coat with
other coats & depth-sorted strokes.

# Comparison with grease pencil

Feedback for grease pencil:

- there's a lack of real "painting" tools that operate on pixels / at the raster level.
    - the issue is that it's vector-based: does not give as much freedom as raster tools
- used in production, 2D and 3D
- rarely used for fur, hair?

https://docs.google.com/spreadsheets/d/199VVlQxMXu5dQkCnx7q__C--vhiQUS9CTr6bDWO1Sxg/edit?gid=1372438798#gid=1372438798

- Lack of "drawing feeling"
- need for a more powerful brush engine
- learning curve too steep
- "raster brushes" (however this can be implemented...)

Possible improvements:

- brush textures
- dynamic lighting
    - fetch/modulate stroke color from shading
    - inherit normals of surface?
      -> paint normals as well
- paint with thickness: strokes put thick "layers" of paint that can stack on top of each other

# Go smaller? Brush strokes put individual "pixels" in 3D space

Instead of generating screen-aligned geometry that represent strokes, strokes now place "3D pixel" on the canvas.
=> i.e. a point cloud sufficiently dense to cover every pixel on the screen.
Brush textures now become volumetric.

Point clouds have been successfully used in Dreams.

Voxel painting, basically? Not really, there's no voxel grid, only points.

Relation to gaussian splatting?

**Extend the raster tools (smudge, smear, etc.) in the third dimension.**
Constrain the spatial range of the raster tools with embedding constraints like overcoat.

Brush strokes lay down points in 3D space, bound to a "bone curve". Points can also blend with nearby strokes.

## The obvious question: how to fill gaps?

# Strokes VS points?

Aka. do we want pixel-level blending tools VS only strokes?

# Move away from Rust for the app?

Reconsider other languages for the application.
Rust has interesting features but the borrow checker forces the dev to make contrived choices sometimes
(main offenders: methods that return a reference to a field lock the whole field)
Honestly, exclusive `&mut` refs are nice in principle, but in practice most of the time I don't care about exclusivity:
it should be OK (at least in single-threaded mode) to have multiple "mutable" (note: not **exclusive**) refs to the
same object. I feel that the only benefit of exclusive refs is to avoid iterator invalidation.

# The way forward: textured strokes

I think it's a mistake to try to port raster tool in 3D (we would need volumetrics).
Instead, embrace strokes (polylines) as primitives and provide tools to work with them efficiently.
Provide a rich model for stroke appearance:

Stroke point attributes:

- point (quantized)
- color
- width
- blur
- arclength

Stroke attributes:

- brush index
- base color

# TODO

- pressure response curve editor (GUI/load/save)
- show stroke paths / points
- visualize attributes on stroke points
- brush image thumbnail

# The main issue: dynamic stamping

https://github.com/ShenCiao/Ciallo: sample stamps in fragment shader, up to a maximum number; no preintegration, but
closed form available for "vanilla" and "airbrush"; for custom brush, resorts to sampling.

How to accelerate? Retain the same appearance as stamping, but with less texture samples, or even only one.
Complications: dynamic variations in stamp size, rotation, opacity... **most likely impossible to preintegrate**

Don't try to simulate everything with stamping. Instead provide specialized models with adjustable parameters:

- airbrush
    - noise

If not enough, do stamping. Also a lot of stamps can be split into multiple airbrush-like stamps.

That said, it would be nice if we were able to preintegrate some anisotropic shapes.
Short-term goal: derive an anisotropic version of the "airbrush" representation in Ciallo.

# Extensibility

`Tool` interface: has access to the scene + current camera, receives gestures, 2D or 3D.

- mouse cursor image
- event handling
- gesture_begin (pos)
- gesture_update (pos)
- gesture_finish
- gesture_cancel
- Processing gestures: call the tool repeatedly

Undo/redo: command-based

## Plugin system?

E.g. custom tools

1. provide a python API (meh, don't like python)
2. rust plugins compiled to WASM
3. rust plugins via C FFI
4. write the app in C#, write plugins in C#

Options (2) or (3) seem the most appropriate, but for (3) run in a separate process.
Go with (2)


# Next steps

- figure out what we want to do
  - "digital painting" but in 3D
  - key point: not texture painting => it has volume and can go outside the silhouette of the object
- ribbon brushes: hide/fade when looking at them from the ends
  - i.e. fade when ribbon normal is perpendicular to view direction
- do we always want ribbons aligned to the screen?
  - no
- consider negative strokes that cut into silhouettes

TODO:
- add normals to curves
- **lights**
- import geometry, and project strokes on it

# Scalability
When drawing close on a surface, should still look good when zooming **out**. Strokes should be properly filtered (anisotropic).


----------------------------------------------

# Testbed

- no "proper" GUI necessary for now, use egui
- most things should be data-driven: touch rust as little as possible
- scene representation should be "tangible": not only b-reps, must be able to query neighbors, ground position, navmeshes, etc.
- lighting
- objects
- scripting: probably lua

## Entities
- name
- parent entity
- file dependencies
- type
  - scene object
  - task
  - variable (accessible in scripts)
  - shader
  
## Scene Objects
- position in scene
- parenting

## Editing Interfaces
- global shader defines
- scene object editor
- import existing assets


## Script interface
- tasks
    - script 
    - state machine
    - native code


# Main loop 

Architecture: main loop that polls async tasks
- single threaded (except for worker threads)
- one global world object, accessible on the main thread only
  - worker threads can't modify the world directly, but they can schedule an update task on the main thread
- tasks have a priority so they can run in a predictable order
  - if two tasks are ready, the one with the higher priority runs first
  - tasks with the same priority class run in an unspecified order, possibly non-
- any task can display things on the screen
   - a display element is tied to the lifetime of the object;
   - i.e. it stays on the screen as long as it lives in the program
- any task can await input events
- any task can show immediate UI (provided it is run on every frame)
- it's always possible to query the current cursor position
- playing a sound just spawns a task (that can be cancelled if necessary)
- entities can be associated to one or more files
  - if the file changes, the entity is reloaded automatically (what this does is up to the entity type)


# World
- serializable (can be saved & reloaded)
  - task state difficult to serialize
  - as little state as possible in tasks
- save/load snapshots for rollback & undo/redo
  - problem: state in tasks makes rollback difficult
  - custom scripting language with serializable state?
- events on entity added / removed / modified, per type

- entity data: slotmap + secondary map for actual data
  - an entity is one thing only, it's *not* a collection of components
- world data (entity data) is pure POD data, no destructors, no dynamic allocation
  - no dynamic allocation means that world data can be serialized easily, and saved/restored with a simple memcpy 
  - big drawbacks: data structures like lists/vectors can't be used, instead they must be replaced by entity lists
  - advantages: entity data can be allocated in a bump allocator
  - big drawback: strings?
     - big strings allocated in a pool, and the rest are just fixed-size strings

- the world is modified in batches
   - each batch records the changes made to the world (entities added/removed, data modified) and stores a copy of the
     original data for rollback
   - systems can then query the changes between two batches: e.g. the rendering system would query the changes made to
     the world since the last frame, and update the rendering data accordingly (create new meshes, textures, etc.)
   - during a batch, modified entities point to the modified data in the tape
   - at the end of the batch, tape data is committed to world data & entities are updated to point to world data
   - nothing references the tape data after the batch is committed, so it can be freed or moved around to the undo tape

# Use cases for async


# Timing
Use for timers:
- sync rendering with vsync
- run game logic at a fixed rate

APIs:
- in async tasks: `delay(duration).await`
- objects: `Timer::new(duration), timer.is_expired()`
- event loop: `request_wakeup`

Low level API: should the low-level API manage individual timers? or just handle the next wakeup time of the event loop (like winit)?
OR: explicitly wait in the event loop, no need for platform-specific timers
  issue: still need platform-specific stuff to sync with the compositor

Goals:
- scheduling a timer should be doable *everywhere* in the code, no need to pass a reference to the event loop around
- yet, if scheduling a timer in some subsystem, only the event loop will be called, it's the responsibility of the  
  handler to dispatch the event to the right subsystem

Tentative API:
- `request_wakeup()` and `LoopEvent::Timeout { target: Instant }` as the only API
   - no need to identify individual timers, just the next wakeup time
   - issue: at each timer wakeup, need to check all timers to see if they are expired
        - the same thing is done in platform-specific code (to store all event deadlines), so there's duplication

Tentative 2:
- `request_timer_wakeup(at) -> TimerToken` and `LoopEvent::Timeout { token: TimerToken }`

Tentative 3: callbacks
- `request_timer_wakeup(callback) -> TimerToken`
- state accessed in the callback must be 


# Pipeline build tool

A command line tool to preprocess shaders
(generate variants according to a set of keywords, and compile them to SPIR-V).
Invokable from build scripts.

Option A: run from build script
Issue: poor error reporting / dev experience if a shader fails to compile. 
       The errors are emitted via cargo:warning=..., no colors, no clickable links, etc.

Option B: separate command line tool
Issue: need to run it by hand


# Terrain rendering

Some kind of LOD for far terrain rendering. 

Insight: for our type of rendering, we mostly care about the details on silhouettes (e.g. ridges of mountains). 
Maybe there's a way to do far-terrain LODs that preserve high-frequency details on silhouettes.

Terrain is fully static, so possible to generate a non-grid mesh that aligns to silhouettes.

Investigate a nanite-like approach for LODs.

Point clouds?

Tentative approach:
- start with a high-res TIN of a terrain heightmap
- slice into chunks
- simplify to 1/4 triangle count
  - for each fine vertex, assign index of nearest coarse triangle for interpolation
- repeat for max LOD count



# Mesh file format

Actually it's more like a memory format.

- position data
- normals (per-vertex)
- attributes (per-vertex)
- indices
- only triangles
- extension: face attributes
- extension: meshlets, meshlet clusters


# Stylization: the important parts

- contours: they should be accurate and robust, not necessarily parametrized, but their width should be modulable via noise. 
   G-buffer based detection is not accurate enough (missing contours when objects with the same normals overlap each other). 
     It may be enough for some primitives, like hair ribbon strokes, when we only care about contours generated by the ribbon twisting on itself.
   Possible solution: scan view space geometry edges for contours; need a special geometry repr, simple triangle meshes not enough to iterate over edges (no adjacency information).  
   For some primitive, like ribbon/swept strokes, it may be possible to extract contours during stroke expansion.

- pixel-precision curves
   Visible discretization of curves into polylines kills the perceived quality of the strokes. They should be subdivided down to individual pixels. This includes contours.