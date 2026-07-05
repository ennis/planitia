use crate::{layout, declare_node};
use crate::layout::LCell;


declare_node! {
    pub static STRUCT: "struct" {
        name   : (str)
        fields : ([FIELD])
    }
    =>
    [I "struct" %name "{" "\n" [V %fields] "}"]
}

declare_node! {
    pub static FIELD: "field" {
        name  :  (str)
        ty    :  (str)
    } => [I %name ":" %ty "\n"]
}

declare_node! {
    pub static STYLE: "style" {}
}

declare_node! {
    pub static RECT: "rect" {
        x        : (num?)
        y        : (num?)
        width    : (num?)
        height   : (num?)
        style    : (@STYLE?)
        children : ([RECT])
    }
    =>
    [I "rect" "(" %x "," %y " " %width "×" %height ")" "{" [V %children] "}"]
}
