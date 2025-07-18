fn main() {
    tauri_build::build();
    println!("cargo:rerun-if-changed=bin/nllb-backend.exe");
}
