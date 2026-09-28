"""Local rule-based insights for video analytics.

Analyzes computed statistics and provides recommendations without requiring
any external API. Uses heuristics based on motion, color diversity, and
activity patterns to suggest the best summarization approach.
"""


def insights_available() -> bool:
    """Local insights are always available - no external dependencies."""
    return True


def generate_insights(summary_stats: dict, summary_context: dict | None = None) -> str:
    """Generate insights using local rule-based analysis.

    Analyzes the video statistics and returns a plain-language interpretation
    with recommendations for summarization technique and length.
    """
    # Extract key metrics with defaults
    total_frames = summary_stats.get('total_frames', 0)
    avg_motion = summary_stats.get('avg_motion', 0)
    max_motion = summary_stats.get('max_motion', 0)
    motion_variance = summary_stats.get('motion_variance', 0)
    color_diversity = summary_stats.get('color_diversity_avg', 0)
    high_activity = summary_stats.get('high_activity_frames', 0)
    low_activity = summary_stats.get('low_activity_frames', 0)

    # Calculate ratios
    activity_ratio = high_activity / max(total_frames, 1)
    stillness_ratio = low_activity / max(total_frames, 1)

    # Classify video type
    video_type = _classify_video_type(avg_motion, motion_variance, activity_ratio, stillness_ratio, color_diversity)

    # Get technique recommendation
    technique, technique_reason = _recommend_technique(video_type, avg_motion, color_diversity, motion_variance)

    # Get length recommendation
    length_rec = _recommend_length(video_type, total_frames, activity_ratio)

    # Build the insight text
    insights = []

    # Video type description
    insights.append(video_type['description'])

    # Key observations
    if avg_motion > 2.0:
        insights.append(f"The video shows significant movement with an average motion score of {avg_motion:.2f}.")
    elif avg_motion < 0.5:
        insights.append(f"This appears to be a relatively static video with minimal camera or subject movement.")

    if color_diversity > 0.1:
        insights.append("There's good visual variety in color and composition across frames.")

    if motion_variance > 5.0:
        insights.append("Motion intensity varies considerably, suggesting distinct active and calm segments.")

    # Recommendations
    insights.append(f"Recommended technique: **{technique}** - {technique_reason}")
    insights.append(length_rec)

    # Context from previous run if available
    if summary_context:
        used_technique = summary_context.get('technique', '')
        compression = summary_context.get('compression_ratio', 0)
        if used_technique and compression:
            if used_technique == technique.lower().replace('-', '_').replace(' ', '_'):
                insights.append(f"Your current settings align well with this recommendation.")
            else:
                insights.append(f"You used {used_technique.replace('_', ' ')} which also works, but {technique.lower()} might capture more relevant content for this video type.")

    return " ".join(insights)


def _classify_video_type(avg_motion, motion_variance, activity_ratio, stillness_ratio, color_diversity):
    """Classify the video into a category based on its characteristics."""

    # High motion + high variance = action video
    if avg_motion > 2.0 and motion_variance > 3.0:
        return {
            'type': 'action',
            'description': "This video contains dynamic, action-heavy content with frequent movement and activity changes."
        }

    # High motion but low variance = continuous motion (sports, driving, etc.)
    if avg_motion > 1.5 and motion_variance < 2.0:
        return {
            'type': 'continuous_motion',
            'description': "This video has consistent, continuous motion throughout - typical of sports footage, driving videos, or tracking shots."
        }

    # Low motion + high color diversity = visual/artistic content
    if avg_motion < 0.8 and color_diversity > 0.08:
        return {
            'type': 'visual',
            'description': "This appears to be visually-focused content with varied scenes but limited motion - possibly a slideshow, presentation, or artistic piece."
        }

    # Very low motion + low color diversity = static content
    if avg_motion < 0.5 and color_diversity < 0.05:
        return {
            'type': 'static',
            'description': "This is a mostly static video, possibly a lecture, interview, or surveillance footage with minimal visual changes."
        }

    # High activity ratio = event-rich content
    if activity_ratio > 0.3:
        return {
            'type': 'event_rich',
            'description': "This video contains numerous distinct events or moments of interest spread throughout its duration."
        }

    # Mixed characteristics
    return {
        'type': 'mixed',
        'description': "This video has mixed characteristics with varying levels of motion and visual diversity across different segments."
    }


def _recommend_technique(video_type, avg_motion, color_diversity, motion_variance):
    """Recommend the best summarization technique based on video type."""

    vtype = video_type['type']

    if vtype == 'action':
        return ('Event-based',
                "captures the key action moments and activity spikes that define this dynamic content")

    if vtype == 'continuous_motion':
        return ('Motion-based',
                "effectively samples from the continuous movement to create a representative summary")

    if vtype == 'visual':
        return ('Color/Histogram-based',
                "preserves the visual diversity and ensures different scenes are represented")

    if vtype == 'static':
        return ('Color-based',
                "identifies the visually distinct moments in otherwise similar frames")

    if vtype == 'event_rich':
        return ('Event-based',
                "detects and preserves the multiple events and highlights throughout the video")

    # Mixed or default
    if motion_variance > 3.0:
        return ('Combined',
                "balances motion detection with visual diversity for varied content like this")

    if color_diversity > 0.08:
        return ('Combined',
                "weighs both motion and color features to capture the full range of content")

    return ('Combined',
            "provides the most balanced approach for videos with mixed characteristics")


def _recommend_length(video_type, total_frames, activity_ratio):
    """Recommend summary length based on video characteristics."""

    vtype = video_type['type']

    if vtype == 'static':
        return "A **short (10%)** summary should be sufficient since there's limited visual variation."

    if vtype == 'action' or vtype == 'event_rich':
        if activity_ratio > 0.4:
            return "Consider a **longer (35%)** summary to preserve the many interesting moments."
        return "A **medium (20%)** summary should capture the key action while keeping it concise."

    if vtype == 'visual':
        return "A **medium (20%)** summary works well to showcase the visual variety without redundancy."

    if vtype == 'continuous_motion':
        return "A **short to medium (10-20%)** summary prevents repetition while showing the motion flow."

    # Default for mixed
    return "A **medium (20%)** summary provides a good balance for this type of content."