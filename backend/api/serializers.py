from rest_framework import serializers


class SummarizeRequestSerializer(serializers.Serializer):
    """Serializer for video summarization request."""

    TECHNIQUE_CHOICES = [
        ('motion', 'Motion-based'),
        ('color', 'Color/Histogram-based'),
        ('event', 'Event-based'),
        ('object_detection', 'Object Detection'),
        ('combined', 'Combined Approach'),
    ]

    OUTPUT_TYPE_CHOICES = [
        ('dynamic', 'Dynamic Video'),
        ('static', 'Static Storyboard'),
        ('both', 'Both'),
    ]

    technique = serializers.ChoiceField(choices=TECHNIQUE_CHOICES)
    output_type = serializers.ChoiceField(choices=OUTPUT_TYPE_CHOICES, default='dynamic')
    frame_sample_rate = serializers.IntegerField(min_value=1, max_value=100, default=2)
    max_frames = serializers.IntegerField(min_value=10, max_value=5000, default=3000)
    output_fps = serializers.IntegerField(min_value=1, max_value=60, default=24)
    summary_percent = serializers.IntegerField(min_value=5, max_value=80, default=20, required=False)

    n_clusters = serializers.IntegerField(min_value=2, max_value=100, default=15, required=False)
    motion_threshold = serializers.FloatField(min_value=0.1, max_value=50.0, default=5.0, required=False)

    motion_weight = serializers.FloatField(min_value=0, max_value=1, default=0.4, required=False)
    color_weight = serializers.FloatField(min_value=0, max_value=1, default=0.3, required=False)
    event_weight = serializers.FloatField(min_value=0, max_value=1, default=0.3, required=False)


class SummaryResultSerializer(serializers.Serializer):
    """Serializer for summarization result."""

    output_path = serializers.CharField(required=False, allow_null=True)
    output_url = serializers.CharField(required=False, allow_null=True)
    storyboard_url = serializers.CharField(required=False, allow_null=True)
    total_frames_processed = serializers.IntegerField()
    key_frames_selected = serializers.IntegerField()
    technique = serializers.CharField()
    output_type = serializers.CharField()
    duration_seconds = serializers.FloatField()
    compression_ratio = serializers.FloatField()
    metadata = serializers.DictField()
    frames = serializers.ListField(child=serializers.DictField(), required=False)


class VideoInfoSerializer(serializers.Serializer):
    """Serializer for video information."""

    filename = serializers.CharField()
    size_bytes = serializers.IntegerField()
    duration_seconds = serializers.FloatField()
    fps = serializers.FloatField()
    width = serializers.IntegerField()
    height = serializers.IntegerField()
    total_frames = serializers.IntegerField()


class TechniqueInfoSerializer(serializers.Serializer):
    """Serializer for technique information."""

    id = serializers.CharField()
    name = serializers.CharField()
    description = serializers.CharField()
    parameters = serializers.ListField(child=serializers.DictField())
