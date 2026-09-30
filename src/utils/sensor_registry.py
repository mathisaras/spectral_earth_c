"""
Sensor Registry - Utility for loading and caching sensor configurations.

Usage:
    from src.utils.sensor_registry import SensorRegistry
    
    # Get a single sensor config
    emit_config = SensorRegistry.get("EMIT")
    
    # Get multiple sensor configs
    configs = SensorRegistry.get_all(["EMIT", "ENMAP", "DESIS"])
"""

from pathlib import Path
from typing import Dict, List, Optional
from omegaconf import OmegaConf


class SensorRegistry:
    """
    Registry for sensor configurations.
    
    Loads sensor YAML configs from configs/sensor/ and caches them.
    Sensor names are case-insensitive (normalized to uppercase internally).
    """
    
    _configs: Dict[str, dict] = {}
    _config_dir: Optional[Path] = None
    
    # Known sensors and their canonical names
    KNOWN_SENSORS = {"EMIT", "ENMAP", "DESIS", "EO1", "S2", "LO", "LT", "S1", "AMMIS", "GAOFEN5"}
    
    @classmethod
    def _get_config_dir(cls) -> Path:
        """Get the path to the sensor configs directory."""
        if cls._config_dir is None:
            # Navigate from src/utils/ to configs/sensor/
            cls._config_dir = Path(__file__).parents[2] / "configs" / "sensor"
        return cls._config_dir
    
    @classmethod
    def get(cls, sensor_name: str) -> dict:
        """
        Get sensor configuration by name.
        
        Args:
            sensor_name: Sensor name (case-insensitive, e.g., "EMIT", "emit", "Emit")
            
        Returns:
            Dictionary containing sensor configuration
            
        Raises:
            FileNotFoundError: If sensor config file doesn't exist
            ValueError: If sensor name is not recognized
        """
        # Normalize to uppercase
        name_upper = sensor_name.upper()
        
        if name_upper not in cls._configs:
            config_path = cls._get_config_dir() / f"{name_upper.lower()}.yaml"
            
            if not config_path.exists():
                available = [f.stem.upper() for f in cls._get_config_dir().glob("*.yaml")]
                raise FileNotFoundError(
                    f"Sensor config not found for '{sensor_name}'. "
                    f"Expected file: {config_path}. "
                    f"Available sensors: {available}"
                )
            
            # Load and cache
            cfg = OmegaConf.load(config_path)
            cls._configs[name_upper] = OmegaConf.to_container(cfg, resolve=True)
        
        return cls._configs[name_upper]
    
    @classmethod
    def get_all(cls, sensor_names: List[str]) -> Dict[str, dict]:
        """
        Get configurations for multiple sensors.
        
        Args:
            sensor_names: List of sensor names (case-insensitive)
            
        Returns:
            Dictionary mapping sensor names (uppercase) to their configs
        """
        return {name.upper(): cls.get(name) for name in sensor_names}
    
    @classmethod
    def get_normalization(cls, sensor_name: str) -> dict:
        """
        Get only the normalization parameters for a sensor.
        
        Args:
            sensor_name: Sensor name (case-insensitive)
            
        Returns:
            Dictionary containing normalization parameters
        """
        config = cls.get(sensor_name)
        return config.get("normalization", {})
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available sensor configurations.
        
        Returns:
            List of sensor names (uppercase) that have config files
        """
        config_dir = cls._get_config_dir()
        return sorted([f.stem.upper() for f in config_dir.glob("*.yaml")])
    
    @classmethod
    def clear_cache(cls):
        """Clear the config cache (useful for testing)."""
        cls._configs.clear()
