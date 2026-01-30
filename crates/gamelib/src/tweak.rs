//! Global tweakable parameters.

use std::any::Any;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::panic::Location;
use std::sync::{LazyLock, Mutex};
use egui::{Ui};
use color::Srgba8;
use math::{Vec2, Vec3, Vec4};

pub trait Tweakable: Any + Send + Sync {
    fn ui(&mut self, ui: &mut egui::Ui, options: &TweakOptions);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

impl Tweakable for f32 {
    fn ui(&mut self, ui: &mut Ui, options: &TweakOptions) {
        if options.range.start().is_infinite() || options.range.end().is_infinite() {
            ui.add(egui::DragValue::new(self).speed(0.1));

        } else {
            ui.add(egui::Slider::new(self, options.range.clone()));
        }
    }
}

impl Tweakable for u32 {
    fn ui(&mut self, ui: &mut Ui, options: &TweakOptions) {
        if options.range.start().is_infinite() || options.range.end().is_infinite() {
            ui.add(egui::DragValue::new(self).speed(1.0));
        } else {
            //ui.add(egui::Slider::new(self, options.range.clone()));
        }
    }
}

impl Tweakable for bool {
    fn ui(&mut self, ui: &mut Ui, _options: &TweakOptions) {
        ui.checkbox(self, "");
    }
}

impl Tweakable for Srgba8 {
    fn ui(&mut self, ui: &mut Ui, _options: &TweakOptions) {
        let mut color = egui::Color32::from_rgba_unmultiplied(self.r, self.g, self.b, self.a);
        if ui.color_edit_button_srgba(&mut color).changed() {
            self.r = color.r();
            self.g = color.g();
            self.b = color.b();
            self.a = color.a();
        }
    }
}

macro_rules! impl_tweakable_vec {
	($ty:ty, [$($field:ident),+]) => {
        impl Tweakable for $ty {
            fn ui(&mut self, ui: &mut Ui, options: &TweakOptions) {
                ui.horizontal(|ui| {
                    $(
                        ui.label(stringify!($field));
                        ui.add(egui::DragValue::new(&mut self.$field).speed(0.1));
                    )+
                });
            }
        }
	}
}

impl_tweakable_vec!(Vec2, [x, y]);
impl_tweakable_vec!(Vec3, [x, y, z]);
impl_tweakable_vec!(Vec4, [x, y, z, w]);

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
pub struct TweakOptions {
    pub range: RangeInclusive<f32> = RangeInclusive::new(-f32::INFINITY, f32::INFINITY),
}

#[track_caller]
pub fn tweak_value<T: Tweakable + Clone>(name: &str, default: T, options: TweakOptions) -> T {
    TWEAKS.lock().unwrap().get_or_insert(Location::caller(), name, default, options).clone()
}

pub fn show_tweaks_gui(ui: &mut Ui) {
    TWEAKS.lock().unwrap().show_gui(ui);
}

#[macro_export]
macro_rules! tweak {
    ($name:ident $(: $ty:ty)? = $default:expr) => {
        {
            let $name $(: $ty)? = $crate::tweak_value(
                stringify!($name),
                $default,
                $crate::TweakOptions { .. },
            );
            $name
        }
    };
}

pub use tweak;

////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
struct Key {
    name: String,
    location: &'static Location<'static>,
}

struct Tweak {
    value: Box<dyn Tweakable>,
    options: TweakOptions,
}

struct Tweaks {
    entries: HashMap<Key, Tweak>,
}

impl Tweaks {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn get_or_insert<T: Tweakable>(
        &mut self,
        location: &'static Location<'static>,
        name: &str,
        default: T,
        options: TweakOptions,
    ) -> &mut T {
        let key = Key {
            name: name.to_string(),
            location,
        };
        if !self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), Tweak {
                value: Box::new(default),
                options,
            });
        }
        let value = &mut *self.entries.get_mut(&key).unwrap().value;
        (value as &mut dyn Any).downcast_mut::<T>().unwrap()
    }

    fn show_gui(&mut self, ui: &mut Ui) {
        egui::Grid::new("tweak_grid").num_columns(2).show(ui, |ui| {
            for (key, tweak) in &mut self.entries {
                ui.label(format!("{}", key.name));
                tweak.value.ui(ui, &tweak.options);
                ui.end_row();
            }
        });
    }
}

static TWEAKS: LazyLock<Mutex<Tweaks>> = LazyLock::new(|| {
    Mutex::new(Tweaks::new())
});