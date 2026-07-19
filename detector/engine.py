import requests
import json
import os

# Global variable to cache the local transformer model pipeline
_local_pipeline = None

def check_api(text: str, platform: str, api_key: str) -> float:
    """
    Sends text to the chosen cloud API (gptzero or sapling) for AI detection.
    
    Returns:
        float: Probability that the text is AI-generated (0.0 to 1.0).
    """
    platform = platform.lower().strip()
    
    if platform == "gptzero":
        url = "https://api.gptzero.me/v2/predict/text"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "document": text,
            "version": "latest"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"GPTZero API error (status code {response.status_code}): {response.text}")
            
        data = response.json()
        documents = data.get("documents", [])
        if not documents:
            raise RuntimeError("GPTZero API returned an empty document analysis list.")
            
        doc = documents[0]
        # Attempt to read completely_generated_prob or document_proba
        ai_score = doc.get("completely_generated_prob")
        if ai_score is None:
            proba = doc.get("document_proba", {})
            ai_score = proba.get("ai")
        if ai_score is None:
            ai_score = doc.get("average_generated_prob")
            
        if ai_score is None:
            raise RuntimeError(f"Could not parse AI score from GPTZero response: {json.dumps(data)}")
            
        return float(ai_score)
        
    elif platform == "sapling":
        url = "https://api.sapling.ai/api/v1/aidetect"
        payload = {
            "key": api_key,
            "text": text,
            "sent_scores": False
        }
        
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Sapling API error (status code {response.status_code}): {response.text}")
            
        data = response.json()
        ai_score = data.get("score")
        if ai_score is None:
            raise RuntimeError(f"Could not parse AI score from Sapling response: {json.dumps(data)}")
            
        return float(ai_score)
        
    else:
        raise ValueError(f"Unsupported API platform: '{platform}'. Supported platforms: gptzero, sapling.")


def chunk_text(text: str, max_chars: int = 1200) -> list:
    """
    Helper to chunk long text into segments under token limits (approx 300 words).
    Splits at paragraph boundaries or punctuation marks when possible.
    """
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if current_length + len(p) > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [p]
            current_length = len(p)
        else:
            current_chunk.append(p)
            current_length += len(p) + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    # If a single chunk is still too long, hard slice it
    sliced_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars * 1.5:
            # Split by sentences or simply by word count
            words = chunk.split()
            words_per_chunk = 200
            for i in range(0, len(words), words_per_chunk):
                sliced_chunks.append(" ".join(words[i:i + words_per_chunk]))
        else:
            sliced_chunks.append(chunk)
            
    return [c for c in sliced_chunks if len(c.strip()) > 10]


def check_local(text: str) -> float:
    """
    Performs local AI detection using Hello-SimpleAI/chatgpt-detector-roberta model.
    Downloads the model if it is not cached locally.
    
    Returns:
        float: Average probability that the text is AI-generated (0.0 to 1.0).
    """
    global _local_pipeline
    
    if _local_pipeline is None:
        # Import transformers inside functions so startup is fast if local checking is not chosen
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "The 'transformers' library is required for offline checking. "
                "Please run: pip install transformers torch"
            )
            
        print("Loading local AI detection model (Hello-SimpleAI/chatgpt-detector-roberta)...")
        print("Note: If this is the first run, it will download ~500MB of model weights. Please wait.")
        
        # Disable progress bar or let it run
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _local_pipeline = pipeline(
            "text-classification", 
            model="Hello-SimpleAI/chatgpt-detector-roberta",
            device=device
        )
        
    chunks = chunk_text(text)
    if not chunks:
        return 0.0
        
    scores = []
    for chunk in chunks:
        # Run classification on chunk
        try:
            results = _local_pipeline(chunk)
            if results:
                res = results[0]
                label = res.get("label") # "Human" or "ChatGPT"
                score = res.get("score", 0.0)
                
                if label == "ChatGPT":
                    scores.append(score)
                elif label == "Human":
                    scores.append(1.0 - score)
                else:
                    # Fallback mapping based on index if labels are LABEL_0 / LABEL_1
                    if "1" in label or "chatgpt" in label.lower():
                        scores.append(score)
                    else:
                        scores.append(1.0 - score)
        except Exception as e:
            # Skip failed chunks (e.g. tokenizer limits) or log them
            continue
            
    if not scores:
        return 0.0
        
    return sum(scores) / len(scores)
