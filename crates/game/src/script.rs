use crate::context::get_context;
use crate::task::{AbortHandle, spawn_abortable};
use log::error;
use mlua::Lua;

pub fn lua() -> &'static Lua {
    &get_context().lua
}

/// Spawns a task from a Lua script.
pub fn spawn_lua_task(code: &str) -> AbortHandle {
    let code = code.to_string();
    spawn_abortable(async move {
        let lua = lua();
        let r = lua.load(code).exec_async().await;
        match r {
            Ok(_) => {}
            Err(err) => {
                error!("Error executing Lua script: {}", err);
            }
        }
    })
}
