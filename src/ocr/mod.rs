pub mod formula;
pub mod layout;
pub mod text;

use ort::{
    ep,
    session::{Session, builder::SessionBuilder},
};

/// 创建统一配置的 SessionBuilder：CUDA 优先（注册失败时静默回退 CPU），开启图优化。
pub(crate) fn session_builder() -> ort::Result<SessionBuilder> {
    Ok(Session::builder()?
        .with_execution_providers([ep::CUDA::default().build(), ep::CPU::default().build()])?
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
