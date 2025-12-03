# utils/logger.py  (last update 03/12/2025)

from __future__ import annotations
import csv
import os
import time
from pathlib import Path
from typing import Dict, Any


class CSVLogger:
    """Simple CSV logger that accumulates training, episode, and eval scalars"""
    
    def __init__(self, out_dir: str | Path, filename: str = "metrics.csv", append: bool = False):
        """Initialize CSV logger
        
        Args:
            out_dir: Output directory (can be str or Path)
            filename: Name of CSV file
            append: If True, append to existing file instead of overwriting
        """
        out_dir = Path(out_dir) if not isinstance(out_dir, Path) else out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        
        self.path = out_dir / filename
        self.append = append
        self.fieldnames = None
        
        # If appending and file exists, read existing headers
        if append and self.path.exists():
            with open(self.path, "r") as f:
                reader = csv.DictReader(f)
                self.fieldnames = reader.fieldnames
            self._file = open(self.path, "a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
            print(f"📝 Appending to existing CSV: {self.path}")
        else:
            self._file = open(self.path, "w", newline="")
            self._writer = None
            print(f"📝 Creating new CSV: {self.path}")
        
        self.start_time = time.time()
    
    def log(self, row: Dict[str, Any]) -> None:
        """Log a row of metrics
        
        Args:
            row: Dictionary of metric name -> value
        """
        # Add timestamp if not present
        row.setdefault("time_sec", time.time() - self.start_time)
        
        # Initialize writer on first log (if not appending)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
            self._writer.writeheader()
        
        # Write row
        self._writer.writerow(row)
        self._file.flush()
    
    def close(self):
        """Close the CSV file"""
        if self._file and not self._file.closed:
            self._file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
