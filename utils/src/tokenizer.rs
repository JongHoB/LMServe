use hf_hub::api::sync::Api;
use tokenizers::tokenizer::{Result, Tokenizer};

pub fn get_tokenizer(tokenizer_name: &str) -> Result<Tokenizer> {
    match Tokenizer::from_pretrained(&tokenizer_name, None) {
        Ok(tokenizer) => Ok(tokenizer),
        Err(_) => {
            let api = Api::new()
                .map_err(|e| format!("Failed to initialize Hugging Face Hub API: {e}"))?;
            let repo = api.model(tokenizer_name.to_string());

            let path = repo.get("tokenizer.json").map_err(|e| {
                format!(
                    "Could not find tokenizer files in repo {}: {}. Tried: tokenizer.json",
                    tokenizer_name, e
                )
            })?;

            Tokenizer::from_file(&path)
                .map_err(|e| format!("Failed to load tokenizer from {:?}: {e}", path).into())
        }
    }
}
