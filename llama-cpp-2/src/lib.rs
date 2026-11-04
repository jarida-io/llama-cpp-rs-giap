//! Bindings to the llama.cpp library.
//!
//! As llama.cpp is a very fast moving target, this crate does not attempt to create a stable API
//! with all the rust idioms. Instead it provided safe wrappers around nearly direct bindings to
//! llama.cpp. This makes it easier to keep up with the changes in llama.cpp, but does mean that
//! the API is not as nice as it could be.
//!
//! # Examples
//!
//! - [simple](https://github.com/utilityai/llama-cpp-rs/tree/main/examples/simple)
//! - [tools](https://github.com/utilityai/llama-cpp-rs/tree/main/examples/tools)
//!
//! # Feature Flags
//!
//! - `cuda` enables CUDA gpu support.
//! - `sampler` adds the [`context::sample::sampler`] struct for a more rusty way of sampling.
use std::ffi::{c_char, CStr, CString, NulError};
use std::fmt::Debug;
use std::num::NonZeroI32;
use std::sync::OnceLock;

use crate::llama_batch::BatchAddError;
use std::os::raw::c_int;
use std::path::PathBuf;
use std::string::FromUtf8Error;

pub mod context;
pub mod gguf;
pub mod llama_backend;
pub mod llama_batch;
#[cfg(feature = "llguidance")]
pub(crate) mod llguidance_sampler;
mod log;
pub mod model;
pub mod openai;
#[cfg(feature = "mtmd")]
pub mod mtmd;
mod ptr;
pub mod sampling;
#[cfg(feature = "common")]
pub mod speculative;
pub mod timing;
pub mod token;
pub mod token_type;
pub mod vocab;

pub use crate::context::session::{LlamaStateSeqFlags, SeqState};

#[cfg(feature = "common")]
pub(crate) fn status_is_ok(status: llama_cpp_sys_2::llama_rs_status) -> bool {
    status == llama_cpp_sys_2::LLAMA_RS_STATUS_OK
}

/// A failable result from a llama.cpp function.
pub type Result<T> = std::result::Result<T, LlamaCppError>;

/// All errors that can occur in the llama-cpp crate.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaCppError {
    /// The backend was already initialized. This can generally be ignored as initializing the backend
    /// is idempotent.
    #[error("BackendAlreadyInitialized")]
    BackendAlreadyInitialized,
    /// There was an error while get the chat template from model.
    #[error("{0}")]
    ChatTemplateError(#[from] ChatTemplateError),
    /// There was an error while decoding a batch.
    #[error("{0}")]
    DecodeError(#[from] DecodeError),
    /// There was an error while encoding a batch.
    #[error("{0}")]
    EncodeError(#[from] EncodeError),
    /// There was an error loading a model.
    #[error("{0}")]
    LlamaModelLoadError(#[from] LlamaModelLoadError),
    /// There was an error creating a new model context.
    #[error("{0}")]
    LlamaContextLoadError(#[from] LlamaContextLoadError),
    /// There was an error adding a token to a batch.
    #[error["{0}"]]
    BatchAddError(#[from] BatchAddError),
    /// see [`EmbeddingsError`]
    #[error(transparent)]
    EmbeddingError(#[from] EmbeddingsError),
    // See [`LlamaSamplerError`]
    /// Backend device not found
    #[error("Backend device {0} not found")]
    BackendDeviceNotFound(usize),
    /// Max devices exceeded
    #[error("Max devices exceeded. Max devices is {0}")]
    MaxDevicesExceeded(usize),
    /// Failed to convert JSON schema to grammar.
    #[cfg(feature = "common")]
    #[error("JsonSchemaToGrammarError: {0}")]
    JsonSchemaToGrammarError(String),
    /// There was an error fitting model parameters to available memory.
    #[cfg(feature = "common")]
    #[error("{0}")]
    FitError(#[from] crate::model::params::FitError),
}

/// There was an error while getting the chat template from a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ChatTemplateError {
    /// gguf has no chat template (by that name)
    #[error("chat template not found - returned null pointer")]
    MissingTemplate,

    /// chat template contained a null byte
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// The chat template was not valid utf8.
    #[error(transparent)]
    Utf8Error(#[from] std::str::Utf8Error),
}

/// Failed fetching metadata value
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MetaValError {
    /// The provided string contains an unexpected null-byte
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// The returned data contains invalid UTF8 data
    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),

    /// Got negative return value. This happens if the key or index queried does not exist.
    #[error("Negative return value. Likely due to a missing index or key. Got return value: {0}")]
    NegativeReturn(i32),
}

/// Failed to Load context
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaContextLoadError {
    /// llama.cpp returned null
    #[error("null reference from llama.cpp")]
    NullReturn,
}

/// Errors from the sequence-state save/restore API.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum StateSeqError {
    /// llama.cpp read or wrote a different number of bytes than expected.
    #[error("state seq size mismatch: expected {expected}, actual {actual}")]
    SizeMismatch {
        /// Number of bytes the caller expected to transfer.
        expected: usize,
        /// Number of bytes llama.cpp actually transferred.
        actual: usize,
    },
}

/// Failed to decode a batch.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecodeError {
    /// No kv cache slot was available.
    #[error("Decode Error 1: NoKvCacheSlot")]
    NoKvCacheSlot,
    /// The number of tokens in the batch was 0.
    #[error("Decode Error -1: n_tokens == 0")]
    NTokensZero,
    /// An unknown error occurred.
    #[error("Decode Error {0}: unknown")]
    Unknown(c_int),
}

/// Failed to decode a batch.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum EncodeError {
    /// No kv cache slot was available.
    #[error("Encode Error 1: NoKvCacheSlot")]
    NoKvCacheSlot,
    /// The number of tokens in the batch was 0.
    #[error("Encode Error -1: n_tokens == 0")]
    NTokensZero,
    /// An unknown error occurred.
    #[error("Encode Error {0}: unknown")]
    Unknown(c_int),
}

/// Fetching embeddings failed.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[error("Embeddings or logits weren't enabled")]
pub struct EmbeddingsError(());

/// Retrieving embeddings sequences failed.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[error("Embeddings or logits weren't enabled, or the model was configured with LLAMA_POOLING_TYPE_NONE")]
pub struct EmbeddingsSeqError(());

/// Errors that can occur when initializing a grammar sampler
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum GrammarError {
    /// The grammar root was not found in the grammar string
    #[error("Grammar root not found in grammar string")]
    RootNotFound,
    /// The trigger word contains null bytes
    #[error("Trigger word contains null bytes")]
    TriggerWordNullBytes,
    /// The grammar string or root contains null bytes
    #[error("Grammar string or root contains null bytes")]
    GrammarNullBytes,
    /// The grammar call returned null
    #[error("Grammar call returned null")]
    NullGrammar,
}

/// Decode a error from llama.cpp into a [`DecodeError`].
impl From<NonZeroI32> for DecodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => DecodeError::NoKvCacheSlot,
            -1 => DecodeError::NTokensZero,
            i => DecodeError::Unknown(i),
        }
    }
}

/// Encode a error from llama.cpp into a [`EncodeError`].
impl From<NonZeroI32> for EncodeError {
    fn from(value: NonZeroI32) -> Self {
        match value.get() {
            1 => EncodeError::NoKvCacheSlot,
            -1 => EncodeError::NTokensZero,
            i => EncodeError::Unknown(i),
        }
    }
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaModelLoadError {
    /// There was a null byte in a provided string and thus it could not be converted to a C string.
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    /// llama.cpp returned a nullptr - this could be many different causes.
    #[error("null result from llama cpp")]
    NullResult,
    /// Failed to convert the path to a rust str. This means the path was not valid unicode
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterInitError {
    /// There was a null byte in a provided string and thus it could not be converted to a C string.
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),
    /// llama.cpp returned a nullptr - this could be many different causes.
    #[error("null result from llama cpp")]
    NullResult,
    /// Failed to convert the path to a rust str. This means the path was not valid unicode
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterSetError {
    /// llama.cpp returned a non-zero error code.
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}

/// An error that can occur when loading a model.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdapterRemoveError {
    /// llama.cpp returned a non-zero error code.
    #[error("error code from llama cpp")]
    ErrorResult(i32),
}

/// get the time (in microseconds) according to llama.cpp
/// ```
/// # use llama_cpp_2::llama_time_us;
/// # use llama_cpp_2::llama_backend::LlamaBackend;
/// let backend = LlamaBackend::init().unwrap();
/// let time = llama_time_us();
/// assert!(time > 0);
/// ```
#[must_use]
pub fn llama_time_us() -> i64 {
    unsafe { llama_cpp_sys_2::llama_time_us() }
}

/// get the max number of devices according to llama.cpp (this is generally cuda devices)
/// ```
/// # use llama_cpp_2::max_devices;
/// let max_devices = max_devices();
/// assert!(max_devices >= 0);
/// ```
#[must_use]
pub fn max_devices() -> usize {
    unsafe { llama_cpp_sys_2::llama_max_devices() }
}

/// is memory mapping supported according to llama.cpp
/// ```
/// # use llama_cpp_2::mmap_supported;
/// let mmap_supported = mmap_supported();
/// if mmap_supported {
///   println!("mmap_supported!");
/// }
/// ```
#[must_use]
pub fn mmap_supported() -> bool {
    unsafe { llama_cpp_sys_2::llama_supports_mmap() }
}

/// is memory locking supported according to llama.cpp
/// ```
/// # use llama_cpp_2::mlock_supported;
/// let mlock_supported = mlock_supported();
/// if mlock_supported {
///    println!("mlock_supported!");
/// }
/// ```
#[must_use]
pub fn mlock_supported() -> bool {
    unsafe { llama_cpp_sys_2::llama_supports_mlock() }
}

/// Convert a JSON schema string into a llama.cpp grammar string.
#[cfg(feature = "common")]
pub fn json_schema_to_grammar(schema_json: &str) -> Result<String> {
    let schema_cstr = CString::new(schema_json)
        .map_err(|err| LlamaCppError::JsonSchemaToGrammarError(err.to_string()))?;
    let mut out = std::ptr::null_mut();
    let rc = unsafe {
        llama_cpp_sys_2::llama_rs_json_schema_to_grammar(schema_cstr.as_ptr(), false, &mut out)
    };

    let result = {
        if !status_is_ok(rc) || out.is_null() {
            return Err(LlamaCppError::JsonSchemaToGrammarError(format!(
                "ffi error {}",
                rc
            )));
        }
        let grammar_bytes = unsafe { CStr::from_ptr(out) }.to_bytes().to_vec();
        let grammar = String::from_utf8(grammar_bytes)
            .map_err(|err| LlamaCppError::JsonSchemaToGrammarError(err.to_string()))?;
        Ok(grammar)
    };

    unsafe { llama_cpp_sys_2::llama_rs_string_free(out) };
    result
}

#[cfg(all(test, feature = "common"))]
mod tests {
    use super::json_schema_to_grammar;
    use crate::model::ChatTemplateResult;
    use std::ffi::{CStr, CString};
    use std::ptr;

    #[test]
    fn json_schema_string_api_returns_grammar() {
        let schema = r#"{
            "type": "object",
            "properties": {
                "city": { "type": "string" },
                "unit": { "enum": ["c", "f"] }
            },
            "required": ["city"]
        }"#;

        let grammar =
            json_schema_to_grammar(schema).expect("string-based schema conversion should succeed");

        assert!(grammar.contains("root ::="));
    }

    #[test]
    fn gemma4_peg_parser_roundtrips_reasoning_and_tool_calls() {
        // Minimal Gemma 4 template containing '<|tool_call>call:' to trigger common_chat_params_init_gemma4
        let gemma4_tmpl = CString::new(
            r#"{#- OpenAI Chat Completions: -#}{% for m in messages %}<|turn>{{ m.role }}
{{ m.content }}<turn|>
{% endfor %}{% if add_generation_prompt %}<|turn>model
<|channel>thought
{% endif %}{# '<|tool_call>call:' #}"#,
        )
        .unwrap();
        let messages = CString::new(r#"[{"role":"user","content":"Weather in Nairobi?"}]"#).unwrap();
        let tools = CString::new(
            r#"[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]"#,
        )
        .unwrap();
        let reasoning_format = CString::new("auto").unwrap();

        let ffi_params = llama_cpp_sys_2::llama_rs_chat_template_oaicompat_params {
            messages: messages.as_ptr(),
            tools: tools.as_ptr(),
            tool_choice: ptr::null(),
            json_schema: ptr::null(),
            grammar: ptr::null(),
            reasoning_format: reasoning_format.as_ptr(),
            chat_template_kwargs: ptr::null(),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: true,
            add_bos: false,
            add_eos: false,
        };
        let mut raw_result = llama_cpp_sys_2::llama_rs_chat_template_result {
            prompt: ptr::null_mut(),
            grammar: ptr::null_mut(),
            parser: ptr::null_mut(),
            generation_prompt: ptr::null_mut(),
            chat_format: 0,
            grammar_lazy: false,
            grammar_triggers: ptr::null_mut(),
            grammar_triggers_count: 0,
            preserved_tokens: ptr::null_mut(),
            preserved_tokens_count: 0,
            additional_stops: ptr::null_mut(),
            additional_stops_count: 0,
        };

        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_apply_chat_template_oaicompat(
                ptr::null(),
                gemma4_tmpl.as_ptr(),
                &ffi_params,
                &mut raw_result,
            )
        };
        assert_eq!(rc, llama_cpp_sys_2::LLAMA_RS_STATUS_OK);
        let parser_str = unsafe { CStr::from_ptr(raw_result.parser) }
            .to_str()
            .unwrap()
            .to_string();
        let gen_prompt = unsafe { CStr::from_ptr(raw_result.generation_prompt) }
            .to_str()
            .unwrap()
            .to_string();
        let chat_format = raw_result.chat_format;
        unsafe { llama_cpp_sys_2::llama_rs_chat_template_result_free(&mut raw_result) };

        assert!(parser_str.starts_with('{'), "parser should be serialized JSON");
        assert_eq!(gen_prompt, "<|turn>model\n<|channel>thought\n");

        let tmpl_res = ChatTemplateResult {
            prompt: String::new(),
            grammar: None,
            grammar_lazy: true,
            grammar_triggers: Vec::new(),
            preserved_tokens: Vec::new(),
            additional_stops: Vec::new(),
            chat_format,
            parser: Some(parser_str),
            generation_prompt: gen_prompt.clone(),
            parse_tool_calls: true,
        };

        let mut stream = tmpl_res.streaming_state_oaicompat().unwrap();
        // Calling update with generation_prompt (as inference_native_tools.rs does) must not double it
        let _ = stream.update(&gen_prompt, true).unwrap();

        let d1 = stream.update("Thinking about Nairobi weather.", true).unwrap();
        let d2 = stream
            .update(
                "<channel|><|tool_call>call:get_weather{location:<|\"|>Nairobi<|\"|>}<tool_call|>",
                false,
            )
            .unwrap();

        let all_deltas: Vec<String> = d1.into_iter().chain(d2).collect();
        let joined = all_deltas.join("\n");
        assert!(
            joined.contains("\"reasoning_content\":\"Thinking about Nairobi weather.\""),
            "expected reasoning_content delta, got: {joined}"
        );
        assert!(
            !joined.contains("\"content\":\"Thinking"),
            "reasoning leaked into content delta: {joined}"
        );
        assert!(
            joined.contains("\"name\":\"get_weather\""),
            "expected tool_calls delta, got: {joined}"
        );
    }
}

/// Failed to apply model chat template.
#[derive(Debug, thiserror::Error)]
pub enum NewLlamaChatMessageError {
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
}

/// Failed to apply model chat template.
#[derive(Debug, thiserror::Error)]
pub enum ApplyChatTemplateError {
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
    /// the string could not be converted to utf8.
    #[error("{0}")]
    FromUtf8Error(#[from] FromUtf8Error),
    /// llama.cpp returned a null pointer for the template result.
    #[error("null result from llama.cpp")]
    NullResult,
    /// llama.cpp returned an error code.
    #[error("ffi error {0}")]
    FfiError(i32),
    /// invalid grammar trigger data returned by llama.cpp.
    #[error("invalid grammar trigger data")]
    InvalidGrammarTriggerType,
}

/// Failed to accept a token in a sampler.
#[derive(Debug, thiserror::Error)]
pub enum SamplerAcceptError {
    /// llama.cpp returned an error code.
    #[error("ffi error {0}")]
    FfiError(i32),
}

/// Get the time in microseconds according to ggml
///
/// ```
/// # use std::time::Duration;
/// # use llama_cpp_2::llama_backend::LlamaBackend;
/// let backend = LlamaBackend::init().unwrap();
/// use llama_cpp_2::ggml_time_us;
///
/// let start = ggml_time_us();
///
/// std::thread::sleep(Duration::from_micros(10));
///
/// let end = ggml_time_us();
///
/// let elapsed = end - start;
///
/// assert!(elapsed >= 10)
#[must_use]
pub fn ggml_time_us() -> i64 {
    unsafe { llama_cpp_sys_2::ggml_time_us() }
}

/// checks if mlock is supported
///
/// ```
/// # use llama_cpp_2::llama_supports_mlock;
///
/// if llama_supports_mlock() {
///   println!("mlock is supported!");
/// } else {
///   println!("mlock is not supported!");
/// }
/// ```
#[must_use]
pub fn llama_supports_mlock() -> bool {
    unsafe { llama_cpp_sys_2::llama_supports_mlock() }
}

/// Backend device type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaBackendDeviceType {
    /// CPU device
    Cpu,
    /// ACCEL device
    Accelerator,
    /// GPU device
    Gpu,
    /// iGPU device
    IntegratedGpu,
    /// Unknown device type
    Unknown,
}

/// A ggml backend device
///
/// The index is can be used from `LlamaModelParams::with_devices` to select specific devices.
#[derive(Debug, Clone)]
pub struct LlamaBackendDevice {
    /// The index of the device
    ///
    /// The index is can be used from `LlamaModelParams::with_devices` to select specific devices.
    pub index: usize,
    /// The name of the device (e.g. "Vulkan0")
    pub name: String,
    /// A description of the device (e.g. "NVIDIA GeForce RTX 3080")
    pub description: String,
    /// The backend of the device (e.g. "Vulkan", "CUDA", "CPU")
    pub backend: String,
    /// Total memory of the device in bytes
    pub memory_total: usize,
    /// Free memory of the device in bytes
    pub memory_free: usize,
    /// Device type
    pub device_type: LlamaBackendDeviceType,
}

/// List ggml backend devices
#[must_use]
pub fn list_llama_ggml_backend_devices() -> Vec<LlamaBackendDevice> {
    let mut devices = Vec::new();
    for i in 0..unsafe { llama_cpp_sys_2::ggml_backend_dev_count() } {
        fn cstr_to_string(ptr: *const c_char) -> String {
            if ptr.is_null() {
                String::new()
            } else {
                unsafe { std::ffi::CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .to_string()
            }
        }
        let dev = unsafe { llama_cpp_sys_2::ggml_backend_dev_get(i) };
        let props = unsafe {
            let mut props = std::mem::zeroed();
            llama_cpp_sys_2::ggml_backend_dev_get_props(dev, &raw mut props);
            props
        };
        let name = cstr_to_string(props.name);
        let description = cstr_to_string(props.description);
        let backend = unsafe { llama_cpp_sys_2::ggml_backend_dev_backend_reg(dev) };
        let backend_name = unsafe { llama_cpp_sys_2::ggml_backend_reg_name(backend) };
        let backend = cstr_to_string(backend_name);
        let memory_total = props.memory_total;
        let memory_free = props.memory_free;
        let device_type = match props.type_ {
            llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_CPU => LlamaBackendDeviceType::Cpu,
            llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_ACCEL => LlamaBackendDeviceType::Accelerator,
            llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_GPU => LlamaBackendDeviceType::Gpu,
            llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_IGPU => LlamaBackendDeviceType::IntegratedGpu,
            _ => LlamaBackendDeviceType::Unknown,
        };
        devices.push(LlamaBackendDevice {
            index: i,
            name,
            description,
            backend,
            memory_total,
            memory_free,
            device_type,
        });
    }
    devices
}

/// Options to configure how llama.cpp logs are intercepted.
#[derive(Default, Debug, Clone)]
pub struct LogOptions {
    disabled: bool,
}

impl LogOptions {
    /// If enabled, logs are sent to tracing. If disabled, all logs are suppressed. Default is for
    /// logs to be sent to tracing.
    #[must_use]
    pub fn with_logs_enabled(mut self, enabled: bool) -> Self {
        self.disabled = !enabled;
        self
    }
}

extern "C" fn logs_to_trace(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const ::std::os::raw::c_char,
    data: *mut ::std::os::raw::c_void,
) {
    // In the "fast-path" (i.e. the vast majority of logs) we want to avoid needing to take the log state
    // lock at all. Similarly, we try to avoid any heap allocations within this function. This is accomplished
    // by being a dummy pass-through to tracing in the normal case of DEBUG/INFO/WARN/ERROR logs that are
    // newline terminated and limiting the slow-path of locks and/or heap allocations for other cases.
    use std::borrow::Borrow;

    let log_state = unsafe { &*(data as *const log::State) };

    if log_state.options.disabled {
        return;
    }

    // If the log level is disabled, we can just return early
    if !log_state.is_enabled_for_level(level) {
        log_state.update_previous_level_for_disabled_log(level);
        return;
    }

    let text = unsafe { std::ffi::CStr::from_ptr(text) };
    let text = text.to_string_lossy();
    let text: &str = text.borrow();

    // As best I can tell llama.cpp / ggml require all log format strings at call sites to have the '\n'.
    // If it's missing, it means that you expect more logs via CONT (or there's a typo in the codebase). To
    // distinguish typo from intentional support for CONT, we have to buffer until the next message comes in
    // to know how to flush it.

    if level == llama_cpp_sys_2::GGML_LOG_LEVEL_CONT {
        log_state.cont_buffered_log(text);
    } else if text.ends_with('\n') {
        log_state.emit_non_cont_line(level, text);
    } else {
        log_state.buffer_non_cont(level, text);
    }
}

/// Redirect llama.cpp logs into tracing.
pub fn send_logs_to_tracing(options: LogOptions) {
    // TODO: Reinitialize the state to support calling send_logs_to_tracing multiple times.

    // We set up separate log states for each backend to make sure that CONT
    // logs between them can't possibly interfere with each other.
    //
    // In other words, if llama.cpp emits a log without a trailing newline and
    // calls a GGML function, the logs won't be weirdly intermixed and instead
    // we'll llama.cpp logs will CONT previous llama.cpp logs and GGML logs
    // will CONT previous ggml logs.
    static LLAMA_STATE: OnceLock<Box<log::State>> = OnceLock::new();
    #[cfg(feature = "mtmd")]
    static MTMD_STATE: OnceLock<Box<log::State>> = OnceLock::new();
    static GGML_STATE: OnceLock<Box<log::State>> = OnceLock::new();

    let llama_heap_state = Box::as_ref(
        LLAMA_STATE
            .get_or_init(|| Box::new(log::State::new(log::Module::LlamaCpp, options.clone()))),
    ) as *const _;

    #[cfg(feature = "mtmd")]
    let mtmd_heap_state = Box::as_ref(
        MTMD_STATE.get_or_init(|| Box::new(log::State::new(log::Module::Mtmd, options.clone()))),
    ) as *const _;

    let ggml_heap_state = Box::as_ref(
        GGML_STATE.get_or_init(|| Box::new(log::State::new(log::Module::GGML, options))),
    ) as *const _;

    unsafe {
        llama_cpp_sys_2::llama_log_set(Some(logs_to_trace), llama_heap_state as *mut _);

        // There is both `mtmd_log_set` and `mtmd_helper_log_set`. We use the
        // helper, since we don't particularly want to differentiate these
        // two, and the helper calls the normal `mtmd_log_set` as well.
        #[cfg(feature = "mtmd")]
        llama_cpp_sys_2::mtmd_helper_log_set(Some(logs_to_trace), mtmd_heap_state as *mut _);

        // GGML has to be set after llama since setting llama logs sets ggml logs as well.
        llama_cpp_sys_2::ggml_log_set(Some(logs_to_trace), ggml_heap_state as *mut _);
    }
}

/// Failed to parse a chat response.
#[derive(Debug, thiserror::Error)]
pub enum ChatParseError {
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
    /// the string could not be converted to utf8.
    #[error("{0}")]
    Utf8Error(#[from] FromUtf8Error),
    /// llama.cpp returned a null pointer for the parse result.
    #[error("null result from llama.cpp")]
    NullResult,
    /// llama.cpp returned an error code.
    #[error("ffi error {0}")]
    FfiError(i32),
}

/// Failed to convert a string to a token sequence.
#[derive(Debug, thiserror::Error)]
pub enum StringToTokenError {
    /// the string contained a null byte and thus could not be converted to a c string.
    #[error("{0}")]
    NulError(#[from] NulError),
    #[error("{0}")]
    /// Failed to convert a provided integer to a [`c_int`].
    CIntConversionError(#[from] std::num::TryFromIntError),
}

/// Failed to convert a token to a string.
#[derive(Debug, thiserror::Error)]
pub enum TokenToStringError {
    /// There was insufficient buffer space to convert the token to a string.
    #[error("Insufficient Buffer Space {0}")]
    InsufficientBufferSpace(std::os::raw::c_int),
    /// The token was not valid utf8.
    #[error("FromUtf8Error {0}")]
    FromUtf8Error(#[from] FromUtf8Error),
}
