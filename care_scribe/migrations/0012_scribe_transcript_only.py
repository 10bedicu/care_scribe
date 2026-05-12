from django.db import migrations, models


def rename_processing_meta_keys_forward(apps, schema_editor):
    """Rename old processing meta keys to the new keys.

    Old -> New:
        provider     -> chat_provider
        audio_model  -> transcribe_model

    Also adds `transcribe_provider` mirroring `chat_provider` so historical
    entries match the new shape.
    """
    Scribe = apps.get_model("care_scribe", "Scribe")
    to_update = []
    for scribe in Scribe.objects.exclude(meta={}).iterator():
        meta = scribe.meta or {}
        processings = meta.get("processings")
        if not processings:
            continue
        changed = False
        for processing in processings:
            if not isinstance(processing, dict):
                continue
            if "provider" in processing and "chat_provider" not in processing:
                processing["chat_provider"] = processing.pop("provider")
                changed = True
            if "audio_model" in processing and "transcribe_model" not in processing:
                processing["transcribe_model"] = processing.pop("audio_model")
                changed = True
            if (
                "chat_provider" in processing
                and "transcribe_provider" not in processing
            ):
                processing["transcribe_provider"] = processing["chat_provider"]
                changed = True
        if changed:
            scribe.meta = meta
            to_update.append(scribe)
            if len(to_update) >= 500:
                Scribe.objects.bulk_update(to_update, ["meta"])
                to_update = []
    if to_update:
        Scribe.objects.bulk_update(to_update, ["meta"])


def rename_processing_meta_keys_reverse(apps, schema_editor):
    """Revert the rename: new keys -> old keys."""
    Scribe = apps.get_model("care_scribe", "Scribe")
    to_update = []
    for scribe in Scribe.objects.exclude(meta={}).iterator():
        meta = scribe.meta or {}
        processings = meta.get("processings")
        if not processings:
            continue
        changed = False
        for processing in processings:
            if not isinstance(processing, dict):
                continue
            if "chat_provider" in processing and "provider" not in processing:
                processing["provider"] = processing.pop("chat_provider")
                changed = True
            if "transcribe_model" in processing and "audio_model" not in processing:
                processing["audio_model"] = processing.pop("transcribe_model")
                changed = True
            if "transcribe_provider" in processing:
                processing.pop("transcribe_provider")
                changed = True
        if changed:
            scribe.meta = meta
            to_update.append(scribe)
            if len(to_update) >= 500:
                Scribe.objects.bulk_update(to_update, ["meta"])
                to_update = []
    if to_update:
        Scribe.objects.bulk_update(to_update, ["meta"])


class Migration(migrations.Migration):

    dependencies = [
        ('care_scribe', '0011_scribefile_mime_type'),
    ]

    operations = [
        migrations.AddField(
            model_name='scribe',
            name='transcript_only',
            field=models.BooleanField(
                default=False,
                help_text='If True, only transcribe the audio without running any AI form-fill processing.',
            ),
        ),
        migrations.RunPython(
            rename_processing_meta_keys_forward,
            rename_processing_meta_keys_reverse,
        ),
    ]
