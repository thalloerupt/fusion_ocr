pub mod formula;
pub mod layout;
pub mod text;

use ort::{
    ep,
    session::{Session, builder::SessionBuilder},
};

/// 可用逻辑核数。
pub(crate) fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

/// 创建统一配置的 CPU SessionBuilder：指定 intra-op 线程数，开启图优化。
pub(crate) fn session_builder(intra_threads: usize) -> ort::Result<SessionBuilder> {
    Ok(Session::builder()?
        .with_execution_providers([ep::CPU::default().build()])?
        .with_intra_threads(intra_threads.max(1))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)?)
}

/// FormulaNet 含图内自回归 Loop，CUDA 每步调度开销较高；实测 CPU 明显更快。
pub(crate) fn cpu_session_builder() -> ort::Result<SessionBuilder> {
    Ok(Session::builder()?
        .with_execution_providers([ep::CPU::default().build()])?
        // 两个 FormulaNet Session 并行时各占一半逻辑核，避免过度订阅。
        .with_intra_threads(8)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)?)
}
