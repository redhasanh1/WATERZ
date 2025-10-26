# Replace ALL Unicode emojis with ASCII equivalents in server_production.py
import re

with open('server_production.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all emojis with ASCII equivalents (COMPLETE LIST)
replacements = {
    '✅': '[OK]',
    '⚠️ ': '[WARNING] ',
    '⚠️': '[WARNING]',
    '🚀': '[RUNNING]',
    '❌': '[ERROR]',
    '⏱️ ': '[TIME] ',
    '⏱️': '[TIME]',
    '📊': '[INFO]',
    '⬇️ ': '[DOWNLOAD] ',
    '⬇️': '[DOWNLOAD]',
    '⬆️ ': '[UPLOAD] ',
    '⬆️': '[UPLOAD]',
    '✂️ ': '[CROP] ',
    '✂️': '[CROP]',
    '🎞️ ': '[ENCODE] ',
    '🎞️': '[ENCODE]',
    '🗑️ ': '[CLEANUP] ',
    '🗑️': '[CLEANUP]',
    '⏭️ ': '[SKIP] ',
    '⏭️': '[SKIP]',
    '🔄': '[RETRY]',
    'ℹ️ ': '[INFO] ',
    'ℹ️': '[INFO]',
    # NEWLY ADDED - Found in recent logs
    '🎬 ': '[SEGMENT] ',
    '🎬': '[SEGMENT]',
    '📐 ': '[CROP] ',
    '📐': '[CROP]',
    '📁 ': '[MASKS] ',
    '📁': '[MASKS]',
    '🎯 ': '[REGEN] ',
    '🎯': '[REGEN]',
    '🎭 ': '[CREATE] ',
    '🎭': '[CREATE]',
    '🎨 ': '[PAINT] ',
    '🎨': '[PAINT]',
    '🔧 ': '[GPU] ',
    '🔧': '[GPU]',
    '🔥 ': '[INIT] ',
    '🔥': '[INIT]',
    '🔍 ': '[POLL] ',
    '🔍': '[POLL]',
    '🧹 ': '[CLEANUP] ',
    '🧹': '[CLEANUP]',
}

for emoji, ascii_equivalent in replacements.items():
    content = content.replace(emoji, ascii_equivalent)

with open('server_production.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] All emojis replaced with ASCII equivalents")
