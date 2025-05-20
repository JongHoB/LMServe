use std::fs;

fn main() {
    fs::create_dir_all("src/pb")
        .unwrap_or_else(|e| panic!("Failed to create directory for pb files {:?}", e));

    tonic_build::configure()
        .build_client(true)
        .build_server(false)
        .out_dir("src/pb")
        .include_file("mod.rs")
        .compile(&["../proto/worker.proto"], &["../proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
}
