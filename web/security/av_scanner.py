"""
Asynchronous antivirus scanner module.
Uses Windows Defender for non-blocking malware scanning.
"""

import os
import subprocess
import threading
import logging
from datetime import datetime
from typing import Callable, Optional

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'security.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('av_scanner')


class AsyncAVScanner:
    """
    Asynchronous antivirus scanner using Windows Defender.
    Scans files in background threads without blocking the main request.
    """

    def __init__(self, quarantine_dir: Optional[str] = None, timeout: int = 30):
        """
        Initialize the scanner.

        Args:
            quarantine_dir: Directory to move infected files (default: web/quarantine)
            timeout: Scan timeout in seconds
        """
        self.quarantine_dir = quarantine_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'quarantine'
        )
        self.timeout = timeout
        os.makedirs(self.quarantine_dir, exist_ok=True)

    def scan_file_async(
        self,
        filepath: str,
        on_threat_callback: Optional[Callable[[str, dict], None]] = None,
        on_clean_callback: Optional[Callable[[str], None]] = None
    ) -> threading.Thread:
        """
        Scan a file asynchronously (non-blocking).

        Args:
            filepath: Path to the file to scan
            on_threat_callback: Called if threat detected (filepath, result_dict)
            on_clean_callback: Called if file is clean (filepath)

        Returns:
            The thread running the scan (for monitoring if needed)
        """
        thread = threading.Thread(
            target=self._scan_worker,
            args=(filepath, on_threat_callback, on_clean_callback),
            daemon=True,
            name=f"av_scan_{os.path.basename(filepath)}"
        )
        thread.start()
        return thread

    def _scan_worker(
        self,
        filepath: str,
        on_threat_callback: Optional[Callable],
        on_clean_callback: Optional[Callable]
    ):
        """Worker thread that performs the actual scan."""
        try:
            result = self._scan_with_defender(filepath)

            if result['threat_found']:
                logger.warning(f"THREAT DETECTED in {filepath}: {result.get('threat_name', 'Unknown')}")
                self._handle_threat(filepath, result)
                if on_threat_callback:
                    on_threat_callback(filepath, result)
            else:
                logger.info(f"File clean: {filepath}")
                if on_clean_callback:
                    on_clean_callback(filepath)

        except Exception as e:
            logger.error(f"AV scan error for {filepath}: {e}")

    def _scan_with_defender(self, filepath: str) -> dict:
        """
        Scan file using Windows Defender (MpCmdRun.exe).

        Returns:
            dict with 'threat_found' (bool) and optionally 'threat_name'
        """
        # Windows Defender command-line scanner location
        defender_paths = [
            r"C:\Program Files\Windows Defender\MpCmdRun.exe",
            r"C:\ProgramData\Microsoft\Windows Defender\Platform\*\MpCmdRun.exe",
        ]

        mp_cmd_run = None
        for path in defender_paths:
            if '*' in path:
                import glob
                matches = glob.glob(path)
                if matches:
                    mp_cmd_run = sorted(matches)[-1]  # Latest version
                    break
            elif os.path.exists(path):
                mp_cmd_run = path
                break

        if not mp_cmd_run:
            logger.warning("Windows Defender not found, skipping scan")
            return {'threat_found': False, 'skipped': True}

        try:
            # Run scan
            # -Scan -ScanType 3 = Custom scan
            # -File = specific file to scan
            # -DisableRemediation = just detect, don't auto-remove (we handle it)
            result = subprocess.run(
                [mp_cmd_run, '-Scan', '-ScanType', '3', '-File', filepath, '-DisableRemediation'],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            # Return code 0 = clean, 2 = threat found
            if result.returncode == 2:
                # Parse output for threat name
                threat_name = "Unknown threat"
                for line in result.stdout.split('\n'):
                    if 'Threat' in line:
                        threat_name = line.strip()
                        break

                return {
                    'threat_found': True,
                    'threat_name': threat_name,
                    'return_code': result.returncode
                }

            return {'threat_found': False, 'return_code': result.returncode}

        except subprocess.TimeoutExpired:
            logger.error(f"AV scan timeout for {filepath}")
            return {'threat_found': False, 'timeout': True}
        except Exception as e:
            logger.error(f"AV scan error: {e}")
            return {'threat_found': False, 'error': str(e)}

    def _handle_threat(self, filepath: str, result: dict):
        """Handle a detected threat by quarantining the file."""
        try:
            if os.path.exists(filepath):
                # Generate quarantine filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = os.path.basename(filepath)
                quarantine_name = f"{timestamp}_{base_name}.quarantine"
                quarantine_path = os.path.join(self.quarantine_dir, quarantine_name)

                # Move to quarantine
                os.rename(filepath, quarantine_path)
                logger.info(f"File quarantined: {filepath} -> {quarantine_path}")

                # Write incident report
                report_path = quarantine_path + '.report.txt'
                with open(report_path, 'w') as f:
                    f.write(f"Threat Report\n")
                    f.write(f"=============\n")
                    f.write(f"Original file: {filepath}\n")
                    f.write(f"Detection time: {datetime.now().isoformat()}\n")
                    f.write(f"Threat name: {result.get('threat_name', 'Unknown')}\n")
                    f.write(f"Quarantine location: {quarantine_path}\n")

        except Exception as e:
            logger.error(f"Failed to quarantine {filepath}: {e}")
            # As a fallback, try to delete the file
            try:
                os.remove(filepath)
                logger.info(f"Threat file deleted: {filepath}")
            except Exception as del_e:
                logger.critical(f"CRITICAL: Could not remove threat file {filepath}: {del_e}")


# Default scanner instance
_default_scanner = None


def get_scanner() -> AsyncAVScanner:
    """Get or create the default scanner instance."""
    global _default_scanner
    if _default_scanner is None:
        _default_scanner = AsyncAVScanner()
    return _default_scanner


def scan_file_async(filepath: str, on_threat: Optional[Callable] = None):
    """Convenience function to scan a file with the default scanner."""
    return get_scanner().scan_file_async(filepath, on_threat_callback=on_threat)
