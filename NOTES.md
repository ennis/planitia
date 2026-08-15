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
  - key point: not texture painting ⇒ it has volume and can go outside the silhouette of the object
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


# Contours: visibility detection

Place each segment generated by contour detection into tiles. 


# GPU API Improvements, 2026

- creating pipelines: copy/pasting functions, update reload_pipelines, add field in App. Should be easier (PIPELINES)
  → Fixed-function pipeline states are now specified in a TOML file and shared across different pipelines

- keeping struct and constants in sync between GLSL & Rust, and also shader interfaces (attachments, arguments) (INTERFACES***)
  → No improvements; there was an experiment where rust-side shader structs were generated from shader code, but it wasn't
    very pleasant to use (additional build step, IDE not updating the generated code, etc.)  

- resizing render targets as the window is resized (RESIZE)
  → added `RenderTarget` type that lazily allocates/reallocates a texture   
  - to add a new render target, must modify three locations
    → still three locations: add a field, add a line in the constructor, add a line in the render function

- allocating and managing temporary render targets (TEMP)
  → no change: not sure that it's worth to do it at such a low level

- setting the viewport and scissors & related state (RENDERSTATE)
  → viewport/scissors are initialized to the size of the render target by default
  → unsure if it's any better, but I can't remember the last time I had an issue related to incorrect viewport state

- allocating render targets with the correct usage (USAGE)
  → the number of usage flags were greatly reduced; also `RenderTarget` includes the `COLOR_ATTACHMENT` usage by default

- to add a new UI option, need to change 3 locations (struct field, struct ctor, UI function) (UI)
  → no changes; there are "tweaks" that can be used to quickly add a controllable parameter anywhere in the app, 
    but they are not very well-developed 

- lists of options are cumbersome to implement in the UI (UI-LISTS)
  → no changes

- making sure that the format of images matches the shader interface; hard to experiment with because of the need to
  update multiple locations (FORMATS)
  → now it's possible to hot-reload fixed-function pipeline states, so at least issues on that front can be fixed
    while the application is running. Identifying such issues is still a problem. 

- samplers should really be defined next to where they are used, i.e. in shaders (SAMPLERS)
  → no changes; even with slang that would be impractical

- more generally: adding stuff is just a lot of copy-paste, making the code unreadable; difficult to abstract because
  unclear about requirements of future algorithms
  - a wrong abstraction costs time if in the future it prevents an algorithm from being implemented efficiently
  → no changes

- reuse vertex or mesh+task shaders (REUSE)
  → now there's a "screen_quad_vertex_shader" that is reused for all full-screen shader passes. Other than that, no improvements.

- managing one-off image view objects is tedious (IMAGE-VIEWS)
  → Image objects now create a default view, which eliminates almost all instances of VkImageView.
    With VK_EXT_descriptor_heap, VkImageViews should disappear altogether. 


General ideas: more hot-reloading, pipeline as data, GUIs, templates, and sane defaults

# Current pain points, 2026

* Keeping sync between shaders and application is still tedious
   * tried a few automatic approaches, nothing satisfying
   * PLAN: the plan is to give up trying, instead focus on approaches with no code on the application side
      * fully declarative rendering pipeline, all in data
* making UI tools is tedious
  * camera controls, click-drag controls, picking
  * PLAN: there should be a framework for that
* iterations are too long
  * shader hot-reload is nice but a lot of stuff is in the application code
    * e.g. mistakes when passing parameters
    * PLAN: make that data driven  
* debugging with renderdoc is tedious
  * renderdoc should be integrated with the application by default
    * it's already the case, but not enabled by default since the perf cost is great
  * PLAN: there should be integrated 
* starting a new experiment is painful
  * too much boilerplate:
    * creating the experiment file
    * add experiment module to mod.rs
    * wire up with main.rs
      * modify render()
      * modify input()
      * modify gui()
      * => three locations to change is *too much*
    * comment out other experiments
    * write code to show open file dialog
    * write code to load data file
    * write the shader
      * copy-paste shaders.toml
    * write the uniform struct
    * copy-paste render pass code
    * write the GUI
    * write interaction code
      * add state for click-drag detection
  * solution:
    * OPT1: write a script that generates all this boilerplate
      * brittle
    * OPT2:
      * give up writing application code
      * make all of this data driven
      * `File > New Project` should be enough
        * creates a basic project with dummy geometry and a simple shader pipeline

* writing GUIs is tedious
  * no design tool, long iteration times
  * PLAN: 
    * there should be a gui design tool; OR
    * GUI description should be hot-reloadable
    * GUI description should contain code, not the opposite (code containing UI description)


# Updating `gamelib::platform`

Multi-window support.
Keep the `APP` singleton.

## Window handlers 

Q: Should inputs be dispatched to the AppHandler or to a window handler?
A: If there's a separate WindowHandler, you need to move or share app state between App and WindowHandler.
   You have to pay the cost of a separate handler object even if there's only one window.

Q: Can AppHandler and WindowHandler be the same object, referenced at two different locations?
A: No, AppHandler would need to be Rc-shared, or WindowHandler would need to be 'static, because AppHandler is assumed to be static right now.

Conclusion: add a "window handle" parameter to the existing AppHandler input and resize functions.

## Opening new windows / window handles

Q: Should the lifetime of the window be tied to the object used to represent it?
A: Depends on what semantics the object has:
  - Rc-shared semantics: no, because it's too easy to stash a reference somewhere and forget it, keeping the window open
  - Unique semantics: should be OK

Q: Do we need to store references to windows in other objects?

If we need to store references to windows in objects (in struct fields), and window objects have unique semantics
then the references are borrows that come with a lifetime parameter, and this becomes intractable very fast
especially in UI contexts where the lifetime of windows isn't very clear. 
Thus: the need to store references implies that window objects have shared reference semantics.

It is desirable to store references to windows. For instance, you can imagine a callback that stores a 
reference to a UI element (common occurrence in kyute). Let's say that this callback is tied to a timer event, 
such that once the timer finished the UI is updated (by calling a method on the UI element).
In turn, the UI element should signal its owner window that it needs to redraw. Thus, the UI element should store a reference to its owner window. 

In some contexts (e.g.: when receiving an input event), we already have a reference to the window that is passed down to us,
and we can use this to control the window.
But this is not true in all contexts (e.g.: timer callbacks are not tied to specific windows). 

Conclusion: we need to store references to windows in other objects. 
So we usually don't refer to windows via `&Window` or `&mut Window`.


Q: how do we open a new window, and what kind of object (with what semantics) do we use to represent it and control it? 
A: Copyable handle.


To create a window, there is `gamelib::create_window() -> WindowHandle`, and to close (destroy) it, there's `gamelib::destroy_window(WindowHandle)`.
This maps to `Platform::create_window()` and `Platform::destroy_window()`.

WindowHandle is either directly a wrapper over the native handle, or, more likely, a slotmap handle.
If the native handle is used often, then it's possible to bundle slotmap index + native handle.
With an exposed slotmap index it's easier to associate per-window data. But maybe this is exposing too much.

Alternatively, add an API to associate arbitrary `dyn Any` userdata to a window.

Window data (e.g.: input state) is completely hidden inside the platform backend, and only accessible via function calls returning a copy of the data (no references).
In practice, the backend is expected to store window data in the platform object instance, or in static variables if that's more convenient. Access from multiple threads is explicitly
not supported: the functions must be called on the main thread (the thread on which the platform backend was created)




# Drawing paint strokes, June 2026

## Blending should be order-independent OR the strokes should be already sorted

Sorting in general is costly. Can't do that every frame.
If strokes are sorted, then it follows that they are *not* drawn in depth order.

## How to render paint strokes

Or, can we do better than just feeding the rasterizer with tessellated strokes?
Assume that the strokes are already sorted (at least within a batch, aka a "coat").

Two implementations:
* HW rasterizer based: tessellate strokes (possibly with mesh shaders), feed them to the rasterizer
  * Advantages
    * AA possible
    * no need for a separate coarse rasterization step
  * Disadvantages
    * Tessellation of fine details / noise can be impractical
    * In practice, tessellation always combined with finer repr (texture, DFs, procedurals)
* Tiling: compute-based coarse rasterize strokes to tiles, then evaluate tiles
  * Advantages:
    * tessellation not necessary, can use distance fields (or other things) in tiles
      * combinations of distance fields / mini-CSG programs?
    * culling possible during coarse raster if stroke is opaque (not sure if this gives an edge over the rasterizer)
    * more/different opportunities for AA?


## Kinds of strokes
- Simple texture splats


# New QoL goal: hot reloading

There are two approaches to make iteration on games & graphics experiments faster:
1. build tools (GUI tools, domain-specific languages, etc.) that turn the experiment into a data-driven project, 
  so that the experiment can be modified without touching the code
2. hot-reload the experiment code

Option 1. is simply too much work. 
For GUIs, I don't find the existing frameworks very pleasant to use. 
Making a GUI for a particular application needs iteration as well, and I don't want to spend time on "making a tool to make GUIs",
because otherwise I will never make anything.
DSLs are not much better: in addition to the compiler/interpreter, you also need tooling (syntax highlighting, LSPs) for it to be practical.
Basically, both options mean working on things not directly related to the experiments, with their own issues and design problems.

Option 2. seems attainable in a more reasonable amount of time. 
However, this raises a lot of architectural questions and goes into the dark corners of rust tooling.
The main challenge is making something that is practical to use, without too much magic and boilerplate, 
and that doesn't require a lot of work to set up for each experiment.

## Hot reloading: what we want

True hot reloading where code changes while the objects in memory stay the same may be too complicated to implement,
and very unsafe.

Another way of thinking about hot-reloading is _persistence_: being able to save as much application state as possible, and restore it after a code change.
Currently, a lot of time is lost between iterations because the application must be put back into the previous state after a code change. 
That's a lot of manual work. Also, a lot of time is lost waiting for recompilation. Thus, there are two axes of improvement:

* (A) Persistence of application state across iterations
* (B) Faster compilation times

Examples of things that are not persisted, but could be:
- camera position: we need to set up the camera correctly on each run
- scene objects (geometry, materials, etc.): we need to browse and open the geometry on each run
- experiment parameters: values of tweakable parameters, like brush size, light position, etc. are not always saved

There are two approaches to persistence: either we make more application state serializable, or we keep the application state in memory
while the code is reloaded. The latter is more difficult to implement, but seems more versatile and less intrusive.

Faster compilation times can be achieved by:
- splitting the project into smaller crates 
  - it's not clear how much this helps, the smaller crates may need to monomorphize a lot of code from libraries anyway
  - also, the crate boundaries may not be clear, and may need to be changed often
- using another compiler backend

## Potential issues

### Static variables
For instance, `gpu` has a static variable that holds the global Vulkan instance. 
The hot-reloaded crate can't have its own copy of it, so it must be dynamically linked somehow.


## Plan for hot-reloading

- `gpu` should be split into `gpu-types` and `gpu`

## Benefits
If hot-reload is robust enough, we can revisit the idea of embedding compiled shaders directly in rust code, 
and generate a type-safe interface to them.

-----------------------------------------------------------------------

Next steps:
- plugin modules should provide/register interfaces
- examples: Renderer
  - takes a scene as input, renders it
- examples: Scene
  - manages scene objects, lights, cameras, etc.
  - does no rendering by itself

Registered objects should be exclusively traits; otherwise plugins would need to depend on each other.

Application is composed of multiple hot-reloadable components, managed by plugins.
Issue: communication between components.
- Components can't directly hold references (borrows) between each other: since they are hot-reloadable and no one owns the
other, the lifetime relationship between them is undefined.
- Components are registered in a central directory. To communicate, acquire a borrow and call a method.
  - Issue: reentrant calls / circular dependencies between components that lead to "already borrowed" issues.

- Alternative: components only communicate by events and read-only shared data.
  - no borrowing issues
  - Drawbacks: harder to debug and reason about

Example: a "scene" component and a "renderer" component

When the scene changes, a global event "SceneChange" is emitted, containing a Rc reference to the scene.
Then, control flows back to the main loop of the host application, and the emitted events are dispatched in sequence.
The renderer receives a copy of this event and updates its internal buffers.

Issue: rendering
Rendering should be a synchronous operation: the host app calls "render(&Image)" on all renderers, and waits for the result.
This is different from other events because it takes a reference to a swapchain image that is only valid within a stack frame,
and control doesn't flow back to the main event loop.

Inconsistency:
- multiple renderer plugins don't make sense without a way to choose which one is used, or a way to order them, at least

Idea: plugins declare their dependencies statically.
When invoking a plugin method, 

-----------------------------------------------------------------------

Plugin interfaces are too complex.
In all likelihood, it makes no sense to have multiple implementations of the same interface trait.
The data structures should live in the application, not in plugins.
Plugins only define behavior, and they don't call other plugins.
Plugins gain access to a "World" object when they are invoked.
They don't expose any internal data to the outside world.

# Unsoundness in gpu submission tickets

            FIXME: this isn't sound, it's possible to retire resources referenced by unsubmitted, but still live, command buffers:
            
            Example:
            - create cmdbuf A, with ticket=8, referencing resource R
            - drop(R) : queue R for deletion, with ticket=8
            - create cmdbuf B, with ticket=9
            - submit(B)
            - after a while, B's completion sets the timeline to 9
                 -> last_completed_submission_index=9
                 -> set completed_ticket = 9
            - queued drop(R) runs because ticket is 8 < 9
            - submit(A)
                 - A references R, but R has been deleted already
            
The main question is: when a resource drops, is it referenced by any live (submitted or not)
command buffers?

The problem: in poll, when running deferred deletion, we assume that all tickets numbers <= max_completed_ticket
have completed, but that is not the case. 
Completed submissions are retired in the order of their ticket numbers, so the above assumption is valid, but 
only for **submitted** command buffer. Crucially, resources last visible by created but not submitted command buffers
are also retired even though the cmdbuf that references them is not yet submitted.

One way to view the problem is that command buffers can be submitted out-of-order, and ticket numbers are useless
to figure out if a command buffer has been executed or not.

Solution: have a separate timeline for tracking resources, 


# Issue: command pools

The current gospel says that there should be one command pool per thread, and each command buffer is allocated
from the pool of the calling thread.
Also, it says that command buffers should not be free individually, but rather the whole pool should be reset.

This is annoying, because we reset pools in device::end_frame, and this is called from the main thread, so we don't
have access to thread-local pools of other threads.
This means that instead of thread_locals, command pools should be stored in a mutex-locked map (thread ID -> pool)
inside the global device, which is ridiculous.

Also, if command buffers are allocated in another thread, they should also be freed in that thread, which is annoying
because currently they are moved in the global device queue on submission, then recycled on the thread that calls
end_frame.

# VK_EXT_descriptor_heap 

How do we map global variables in slang to PushData?

