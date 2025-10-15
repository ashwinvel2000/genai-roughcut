"""
Pipeline module: orchestrates the full flow.
Script (Template/Transformers/OpenAI) -> TTS (gTTS/ElevenLabs) -> Images (Pexels/Placeholder) -> Assemble (MoviePy)
"""

import logging
import json
import os
import asyncio
from typing import Dict, Any, List
from pathlib import Path
import requests
from gtts import gTTS
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
from PIL import Image, ImageDraw, ImageFont
import re

logger = logging.getLogger(__name__)

# Cache for transformer model
_transformer_model = None
_transformer_tokenizer = None


def _load_transformer_model():
    """Load and cache DistilGPT-2 model."""
    global _transformer_model, _transformer_tokenizer
    
    if _transformer_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info("Loading DistilGPT-2 model...")
            _transformer_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            _transformer_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            _transformer_tokenizer.pad_token = _transformer_tokenizer.eos_token
            logger.info("Model loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    return _transformer_tokenizer, _transformer_model


def script_from_template(brief: str) -> List[str]:
    """Generate 5 deterministic lines from template."""
    logger.info("Generating script from template...")
    words = re.findall(r'\b\w+\b', brief.lower())
    topic = ' '.join(words[:3]) if words else 'this topic'
    
    return [
        f"Welcome to our exploration of {topic}.",
        f"Let's dive into what makes {topic} so fascinating.",
        f"Here are the key insights about {topic}.",
        f"Understanding {topic} can transform your perspective.",
        f"Thanks for joining us on this journey through {topic}."
    ]


def script_with_transformer(brief: str) -> List[str]:
    """
    Generate script with DistilGPT-2 using structured prompt engineering.
    
    Strategy: Generate 5 separate completions for different aspects of the topic,
    rather than one long text. This produces cleaner, more varied results.
    """
    logger.info("Generating script with DistilGPT-2...")
    
    try:
        import torch
        tokenizer, model = _load_transformer_model()
        
        # Extract main topic
        words = re.findall(r'\b\w+\b', brief.lower())
        topic = ' '.join(words[:3]) if len(words) >= 3 else brief
        
        # Five different prompts for variety (intro, explanation, example, impact, conclusion)
        prompts = [
            f"{topic.title()} is",
            f"How {topic} works:",
            f"{topic.title()} can be used to",
            f"The future of {topic} will",
            f"In summary, {topic}"
        ]
        
        lines = []
        
        for i, starter in enumerate(prompts):
            try:
                inputs = tokenizer(starter, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=40,  # Shorter for cleaner output
                        num_return_sequences=1,
                        temperature=0.7,  # Slightly lower for coherence
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2  # Reduce repetition
                    )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the generated text
                # Remove the starter if it appears
                generated = generated.strip()
                
                # Extract first complete sentence
                sentences = re.split(r'[.!?]+', generated)
                if sentences and sentences[0].strip():
                    line = sentences[0].strip()
                    
                    # Validate: must be reasonable length and have actual content
                    word_count = len(line.split())
                    if word_count >= 5 and word_count <= 20 and not line.startswith('Part'):
                        # Add period if missing
                        if not line.endswith(('.', '!', '?')):
                            line += '.'
                        lines.append(line)
                    else:
                        # Use template fallback for this line
                        logger.warning(f"Generated line {i+1} invalid, using template")
                        lines.append(script_from_template(brief)[i])
                else:
                    # Fallback to template
                    lines.append(script_from_template(brief)[i])
                    
            except Exception as e:
                logger.warning(f"Error generating line {i+1}: {e}")
                lines.append(script_from_template(brief)[i])
        
        # Ensure we have exactly 5 lines
        while len(lines) < 5:
            lines.append(script_from_template(brief)[len(lines)])
        
        logger.info(f"✓ Generated {len(lines)} lines with transformer")
        return lines[:5]
        
    except Exception as e:
        logger.error(f"Transformer failed completely: {e}")
        return script_from_template(brief)


def script_with_openai(brief: str, settings) -> List[str]:
    """Generate script with OpenAI API."""
    logger.info("Generating script with OpenAI...")
    
    if not settings.openai_api_key:
        logger.warning("No OpenAI key, falling back to transformer")
        return script_with_transformer(brief)
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a video script writer. Create exactly 5 short, engaging lines for a 30-second video. Each line should be concise and narration-ready."},
                {"role": "user", "content": f"Brief: {brief}"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        text = response.choices[0].message.content.strip()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        lines = [re.sub(r'^\d+[\.\)]\s*', '', line) for line in lines]
        
        if len(lines) >= 5:
            return lines[:5]
        else:
            template = script_from_template(brief)
            return lines + template[len(lines):5]
            
    except Exception as e:
        logger.error(f"OpenAI failed: {e}")
        return script_with_transformer(brief)


def generate_script(brief: str, settings) -> List[str]:
    """Route to appropriate script generation method."""
    mode = settings.script_mode.lower()
    
    if mode == "template":
        return script_from_template(brief)
    elif mode == "transformers":
        return script_with_transformer(brief)
    elif mode == "openai":
        return script_with_openai(brief, settings)
    else:
        logger.warning(f"Unknown script_mode: {mode}, using transformers")
        return script_with_transformer(brief)


def synth_gtts(text: str, output_path: Path):
    """Synthesize speech with gTTS."""
    logger.info(f"Synthesizing with gTTS: {output_path.name}")
    
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(str(output_path))
    logger.info(f"Saved {output_path.name}")


def synth_elevenlabs(text: str, output_path: Path, settings):
    """Synthesize with ElevenLabs API."""
    logger.info(f"Synthesizing with ElevenLabs: {output_path.name}")
    
    if not settings.elevenlabs_api_key:
        logger.warning("No ElevenLabs key, falling back to gTTS")
        synth_gtts(text, output_path)
        return
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": settings.elevenlabs_api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        
        output_path.write_bytes(response.content)
        logger.info(f"Saved {output_path.name}")
        
    except Exception as e:
        logger.error(f"ElevenLabs failed: {e}, falling back to gTTS")
        synth_gtts(text, output_path)


def synthesize_audio(text: str, output_path: Path, settings):
    """Route to appropriate TTS method."""
    mode = settings.tts_mode.lower()
    
    if mode == "gtts":
        synth_gtts(text, output_path)
    elif mode == "elevenlabs":
        synth_elevenlabs(text, output_path, settings)
    else:
        logger.warning(f"Unknown tts_mode: {mode}, using gTTS")
        synth_gtts(text, output_path)


def extract_visual_keywords(text: str) -> str:
    """
    Extract visual keywords from script line for better image generation.
    Removes filler words and focuses on nouns/adjectives for visual concepts.
    
    Args:
        text: Script line
        
    Returns:
        Visual prompt suitable for image generation
    """
    # Common filler words to remove
    filler_words = {
        'welcome', 'to', 'our', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'for',
        'this', 'that', 'these', 'those', 'let', 'lets', 'here', 'there', 'what', 'how', 'why',
        'can', 'could', 'should', 'would', 'will', 'may', 'might', 'must', 'is', 'are', 'was',
        'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'thanks', 'thank',
        'you', 'your', 'us', 'we', 'join', 'joining', 'journey', 'through', 'about', 'into',
        'understanding', 'exploring', 'exploration', 'dive', 'so', 'very', 'really', 'just'
    }
    
    # Tokenize and filter
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if w not in filler_words and len(w) > 2]
    
    # Take first 5 meaningful keywords
    if keywords:
        return ' '.join(keywords[:5])
    else:
        # Fallback to original text if no keywords found
        return ' '.join(text.split()[:5])


def pollinations_image(prompt: str, output_path: Path, width: int = 1280, height: int = 720, max_retries: int = 3) -> bool:
    """
    Generate AI image using Pollinations.ai (100% free, no API key).
    
    Pollinations.ai provides unlimited free AI image generation via simple URL API.
    
    Args:
        prompt: Text description of the image
        output_path: Where to save the generated image
        width: Image width (default: 1280)
        height: Image height (default: 720)
        max_retries: Number of retry attempts (default: 3)
        
    Returns:
        True if successful, False otherwise
    """
    import urllib.parse
    import time
    
    # Truncate prompt if too long (Pollinations works better with shorter prompts)
    if len(prompt) > 200:
        prompt = prompt[:200]
    
    logger.info(f"Generating image with Pollinations.ai: {prompt[:70]}...")
    
    # URL-encode the prompt
    encoded_prompt = urllib.parse.quote(prompt)
    
    # Pollinations.ai API endpoint
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"
    
    for attempt in range(max_retries):
        try:
            # Fetch the image
            response = requests.get(url, timeout=45)  # Increased timeout
            response.raise_for_status()
            
            # Validate response
            if len(response.content) < 1000:  # Too small, likely an error
                raise ValueError(f"Response too small: {len(response.content)} bytes")
            
            # Save to file
            output_path.write_bytes(response.content)
            logger.info(f"✓ Pollinations.ai image saved: {output_path.name} ({len(response.content)/1024:.1f}KB)")
            return True
            
        except Exception as e:
            logger.warning(f"Pollinations.ai attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
    
    logger.error(f"Pollinations.ai failed after {max_retries} attempts")
    return False


def placeholder_image(width: int, height: int, text: str, output_path: Path):
    """Create placeholder image with Pillow."""
    logger.info(f"Creating placeholder: {output_path.name}")
    
    import random
    colors = [
        (100, 149, 237),  # Cornflower blue
        (255, 182, 193),  # Light pink
        (152, 251, 152),  # Pale green
        (255, 218, 185),  # Peach
        (221, 160, 221),  # Plum
    ]
    
    bg_color = random.choice(colors)
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, fill=(255, 255, 255), font=font)
    
    img.save(str(output_path))
    logger.info(f"Saved {output_path.name}")


def fetch_pexels_image(query: str, output_path: Path, settings) -> bool:
    """Fetch image from Pexels API."""
    if not settings.pexels_api_key:
        logger.warning("No Pexels key")
        return False
    
    logger.info(f"Fetching Pexels image for: {query}")
    
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": settings.pexels_api_key}
    params = {"query": query, "per_page": 1, "orientation": "landscape"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get("photos"):
            photo_url = data["photos"][0]["src"]["large"]
            img_response = requests.get(photo_url, timeout=10)
            img_response.raise_for_status()
            
            output_path.write_bytes(img_response.content)
            logger.info(f"Saved {output_path.name}")
            return True
        else:
            logger.warning(f"No Pexels results for: {query}")
            return False
            
    except Exception as e:
        logger.error(f"Pexels fetch failed: {e}")
        return False


def get_or_create_image(query: str, output_path: Path, settings, brief: str = "", scene_num: int = 0):
    """
    Get image based on configured IMAGE_MODE setting.
    
    Modes:
    - "pollinations": AI-generated images (free, unlimited)
    - "pexels": Stock photos (requires API key)
    - "pillow": Simple placeholders
    
    Args:
        query: Raw script line text
        output_path: Where to save the image
        settings: Application settings
        brief: Original user brief for context (e.g., "artificial intelligence basics")
        scene_num: Scene number (0-4) for aspect variation
        
    Returns:
        str: Method used ("pollinations", "pexels", or "placeholder")
    """
    # Scene aspects to add variety
    scene_aspects = [
        "introduction overview concept",      # Scene 1: Opening
        "technical details close-up",         # Scene 2: Deep dive
        "practical applications real-world",  # Scene 3: Use cases
        "future possibilities innovation",    # Scene 4: Forward-looking
        "summary conclusion key-points"       # Scene 5: Wrap-up
    ]
    
    # Extract visual keywords from query
    visual_prompt = extract_visual_keywords(query)
    
    # Simplify prompt for better success rate
    # Format: "keywords, aspect, style" (shorter is better)
    if brief and scene_num < len(scene_aspects):
        # Combine keywords + aspect (don't duplicate brief)
        enhanced_prompt = f"{visual_prompt}, {scene_aspects[scene_num]}, digital art"
    else:
        enhanced_prompt = f"{visual_prompt}, digital art"
    
    logger.info(f"Scene {scene_num+1} prompt: '{enhanced_prompt}' (mode: {settings.image_mode})")
    
    # Use configured image mode
    if settings.image_mode == "pexels":
        # Try Pexels first if explicitly requested
        if settings.pexels_api_key:
            success = fetch_pexels_image(enhanced_prompt, output_path, settings)
            if success:
                return "pexels"
            logger.warning("Pexels failed, falling back to Pollinations")
            success = pollinations_image(enhanced_prompt, output_path)
            if success:
                return "pollinations"
        else:
            logger.warning("Pexels mode selected but no API key - using Pollinations")
            success = pollinations_image(enhanced_prompt, output_path)
            if success:
                return "pollinations"
    
    elif settings.image_mode == "pillow":
        # Use placeholder directly
        logger.info(f"Using Pillow placeholder for: {enhanced_prompt}")
        placeholder_image(1280, 720, enhanced_prompt, output_path)
        return "placeholder"
    
    else:  # Default: pollinations
        # Try Pollinations first
        success = pollinations_image(enhanced_prompt, output_path)
        if success:
            return "pollinations"
        
        # Fallback to Pexels (if API key available)
        if settings.pexels_api_key:
            logger.warning("Pollinations failed, trying Pexels")
            success = fetch_pexels_image(enhanced_prompt, output_path, settings)
            if success:
                return "pexels"
    
    # Final fallback: Pillow placeholder
    logger.info(f"All image sources failed, using Pillow placeholder for: {enhanced_prompt}")
    placeholder_image(1280, 720, enhanced_prompt, output_path)
    return "placeholder"


def assemble_video(
    script_lines: List[str],
    voice_path: Path,
    image_paths: List[Path],
    output_path: Path
):
    """Assemble final MP4 with MoviePy."""
    logger.info("Assembling video...")
    
    audio = AudioFileClip(str(voice_path))
    total_duration = audio.duration
    scene_duration = total_duration / len(script_lines)
    
    clips = []
    for i, (line, img_path) in enumerate(zip(script_lines, image_paths)):
        start = i * scene_duration
        end = (i + 1) * scene_duration
        
        img_clip = ImageClip(str(img_path), duration=scene_duration)
        
        txt_clip = TextClip(
            text=line,
            font_size=32,
            color='white',
            bg_color='black',
            size=(1200, None),
            method='caption'
        ).with_duration(scene_duration).with_position(('center', 'bottom'))
        
        composite = CompositeVideoClip([img_clip, txt_clip])
        clips.append(composite)
    
    final = concatenate_videoclips(clips, method="compose")
    final = final.with_audio(audio)
    
    final.write_videofile(
        str(output_path),
        fps=24,
        codec='libx264',
        audio_codec='aac',
        preset='medium',
        logger=None
    )
    
    audio.close()
    final.close()
    for clip in clips:
        clip.close()
    
    logger.info(f"Video saved: {output_path.name}")


def process_run(run_id: str, brief: str, settings, runs: Dict, storage):
    """Main pipeline orchestration."""
    logger.info(f"Starting run {run_id}")
    
    try:
        # Ensure run directory exists
        storage.ensure_run_dirs(run_id)
        
        # Step 1: Generate script
        runs[run_id]["step"] = "script"
        runs[run_id]["message"] = "🎬 Generating script..."
        logger.info(f"[{run_id}] Generating script...")
        
        script_lines = generate_script(brief, settings)
        script_path = Path(storage.get_script_path(run_id))
        script_path.write_text(json.dumps(script_lines, indent=2))
        logger.info(f"Script saved: {len(script_lines)} lines")
        
        # Step 2: Text-to-speech
        runs[run_id]["step"] = "tts"
        runs[run_id]["message"] = "🎙️ Synthesizing voiceover..."
        logger.info(f"[{run_id}] Synthesizing audio...")
        
        full_text = ' '.join(script_lines)
        voice_path = Path(storage.get_voice_path(run_id))
        synthesize_audio(full_text, voice_path, settings)
        logger.info(f"Audio saved: {voice_path.name}")
        
        # Step 3: Fetch/create images
        runs[run_id]["step"] = "images"
        runs[run_id]["message"] = "📸 Creating images (0/5)..."
        logger.info(f"[{run_id}] Creating images...")
        
        image_paths = []
        for i, line in enumerate(script_lines):
            runs[run_id]["message"] = f"📸 Creating images ({i+1}/5)..."
            img_path = Path(storage.get_image_path(run_id, i))
            # Pass full script line, brief, and scene number for better context
            get_or_create_image(line, img_path, settings, brief=brief, scene_num=i)
            image_paths.append(img_path)
            logger.info(f"Image {i+1}/5 created")
        
        # Step 4: Assemble video
        runs[run_id]["step"] = "assemble"
        runs[run_id]["message"] = "🎥 Assembling video (this may take 30-60 seconds)..."
        logger.info(f"[{run_id}] Assembling video...")
        
        output_path = Path(storage.run_dir(run_id)) / "output.mp4"
        assemble_video(script_lines, voice_path, image_paths, output_path)
        logger.info(f"Video assembled: {output_path.name}")
        
        # Step 5: Complete
        runs[run_id]["step"] = "done"
        runs[run_id]["done"] = True
        runs[run_id]["message"] = "✅ Video generated successfully!"
        runs[run_id]["video_url"] = f"/outputs/{run_id}/output.mp4"
        
        # Save log
        log_path = Path(storage.run_dir(run_id)) / "log.json"
        log_path.write_text(json.dumps({
            "run_id": run_id,
            "brief": brief,
            "script_lines": script_lines,
            "status": "done"
        }, indent=2))
        
        logger.info(f"Run {run_id} complete")
        
    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}", exc_info=True)
        runs[run_id]["step"] = "error"
        runs[run_id]["done"] = False
        runs[run_id]["error"] = str(e)
        runs[run_id]["message"] = f"❌ Error: {str(e)}"


# ============================================================================
# ASYNC PIPELINE - For WebSocket Real-time Updates
# ============================================================================

async def generate_script_async(brief: str, settings) -> List[str]:
    """Async wrapper for script generation (CPU-bound, run in thread pool)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_script, brief, settings)


async def synthesize_audio_async(text: str, output_path: Path, settings):
    """Async wrapper for TTS (I/O-bound, run in thread pool)."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, synthesize_audio, text, output_path, settings)


async def get_or_create_image_async(query: str, output_path: Path, settings, brief: str = "", scene_num: int = 0):
    """Async wrapper for image creation (I/O-bound, run in thread pool).
    
    Returns:
        str: Method used ("pollinations", "pexels", or "placeholder")
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_or_create_image, query, output_path, settings, brief, scene_num)


async def assemble_video_async(script_lines: List[str], voice_path: Path, 
                               image_paths: List[Path], output_path: Path):
    """Async wrapper for video assembly (CPU/I/O-bound, run in thread pool)."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, assemble_video, script_lines, voice_path, image_paths, output_path)


async def process_run_async(run_id: str, brief: str, settings, runs: Dict, storage, websocket=None):
    """
    Async pipeline orchestration with real-time WebSocket updates.
    
    Args:
        run_id: Unique run identifier
        brief: User's video concept brief
        settings: Application settings
        runs: Shared runs dictionary
        storage: Storage helper module
        websocket: Optional WebSocket for real-time updates
    """
    logger.info(f"Starting async run {run_id}")
    
    async def send_update(data: Dict):
        """Helper to send WebSocket updates."""
        if websocket:
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket update: {e}")
    
    try:
        # Ensure run directory exists
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, storage.ensure_run_dirs, run_id)
        
        # Step 1: Generate script
        runs[run_id]["step"] = "script"
        runs[run_id]["message"] = "Generating script..."
        await send_update({"step": "script", "progress": 20, "message": "Generating script..."})
        logger.info(f"[{run_id}] Generating script...")
        
        script_lines = await generate_script_async(brief, settings)
        script_path = Path(storage.get_script_path(run_id))
        await loop.run_in_executor(None, script_path.write_text, json.dumps(script_lines, indent=2))
        logger.info(f"Script saved: {len(script_lines)} lines")
        
        # Send each script line to frontend
        for idx, line in enumerate(script_lines, start=1):
            await send_update({"type": "script_line", "line": line, "index": idx})
        
        # Step 2: Text-to-speech
        runs[run_id]["step"] = "tts"
        runs[run_id]["message"] = "Synthesizing voiceover..."
        await send_update({"step": "tts", "progress": 40, "message": "Synthesizing voiceover..."})
        logger.info(f"[{run_id}] Synthesizing audio...")
        
        full_text = ' '.join(script_lines)
        voice_path = Path(storage.get_voice_path(run_id))
        await synthesize_audio_async(full_text, voice_path, settings)
        logger.info(f"Audio saved: {voice_path.name}")
        
        # Step 3: Fetch/create images with RATE LIMITING
        runs[run_id]["step"] = "images"
        runs[run_id]["message"] = "Creating images (0/5)..."
        await send_update({"step": "images", "progress": 50, "message": "Creating images (0/5)..."})
        logger.info(f"[{run_id}] Creating images with rate limiting...")
        
        # Create semaphore for rate limiting (max 2 concurrent requests to Pollinations)
        semaphore = asyncio.Semaphore(2)
        
        async def rate_limited_image(semaphore, line, img_path, settings, brief, scene_num):
            """Rate-limited image generation."""
            async with semaphore:
                result = await get_or_create_image_async(line, img_path, settings, brief=brief, scene_num=scene_num)
                # Notify frontend when image is ready
                method = "AI" if result == "pollinations" else "Pexels" if result == "pexels" else "Placeholder"
                await send_update({"type": "image_ready", "scene_num": scene_num, "method": method})
                return result
        
        # Create all image tasks with brief context and scene numbers
        image_tasks = []
        image_paths = []
        for i, line in enumerate(script_lines):
            img_path = Path(storage.get_image_path(run_id, i))
            # Pass full script line, brief, and scene number for better context
            image_tasks.append(rate_limited_image(semaphore, line, img_path, settings, brief, i))
            image_paths.append(img_path)
        
        # Run image generations with rate limiting (max 2 at a time)
        for i, task in enumerate(asyncio.as_completed(image_tasks)):
            await task
            progress = 50 + int((i + 1) / len(image_tasks) * 10)  # 50-60%
            runs[run_id]["message"] = f"Creating images ({i+1}/5)..."
            await send_update({"step": "images", "progress": progress, "message": f"Creating images ({i+1}/5)..."})
            logger.info(f"Image {i+1}/5 created")
        
        # Step 4: Assemble video
        runs[run_id]["step"] = "assemble"
        runs[run_id]["message"] = "Assembling video (this may take 30-60 seconds)..."
        await send_update({"step": "assemble", "progress": 80, "message": "Assembling video (this may take 30-60 seconds)..."})
        logger.info(f"[{run_id}] Assembling video...")
        
        output_path = Path(storage.run_dir(run_id)) / "output.mp4"
        await assemble_video_async(script_lines, voice_path, image_paths, output_path)
        logger.info(f"Video assembled: {output_path.name}")
        
        # Step 5: Complete
        runs[run_id]["step"] = "done"
        runs[run_id]["done"] = True
        runs[run_id]["message"] = "Video generated successfully!"
        runs[run_id]["video_url"] = f"/outputs/{run_id}/output.mp4"
        
        await send_update({
            "step": "done", 
            "progress": 100, 
            "message": "Video generated successfully!",
            "video_url": runs[run_id]["video_url"],
            "done": True
        })
        
        # Save log
        log_path = Path(storage.run_dir(run_id)) / "log.json"
        log_data = json.dumps({
            "run_id": run_id,
            "brief": brief,
            "script_lines": script_lines,
            "status": "done"
        }, indent=2)
        await loop.run_in_executor(None, log_path.write_text, log_data)
        
        logger.info(f"Run {run_id} complete")
        
    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}", exc_info=True)
        runs[run_id]["step"] = "error"
        runs[run_id]["done"] = False
        runs[run_id]["error"] = str(e)
        runs[run_id]["message"] = f"Error: {str(e)}"
        
        await send_update({
            "step": "error",
            "progress": 100,
            "message": f"Error: {str(e)}",
            "done": True,
            "error": str(e)
        })
