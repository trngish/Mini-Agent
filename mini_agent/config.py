"""Configuration management module

Provides unified configuration loading and management functionality with support for:
- YAML configuration files
- Environment variable overrides
- CLI argument merging
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    """Retry configuration"""

    enabled: bool = True
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class LLMConfig(BaseModel):
    """LLM configuration"""

    api_key: str
    api_base: str = "https://api.minimax.io"
    model: str = "MiniMax-M2.5"
    provider: str = "anthropic"  # "anthropic" or "openai"
    retry: RetryConfig = Field(default_factory=RetryConfig)


class AgentConfig(BaseModel):
    """Agent configuration"""

    max_steps: int = 50
    workspace_dir: str = "./workspace"
    system_prompt_path: str = "system_prompt.md"


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) timeout configuration"""

    connect_timeout: float = 10.0  # Connection timeout (seconds)
    execute_timeout: float = 60.0  # Tool execution timeout (seconds)
    sse_read_timeout: float = 120.0  # SSE read timeout (seconds)


class ToolsConfig(BaseModel):
    """Tools configuration"""

    # Basic tools (file operations, bash)
    enable_file_tools: bool = True
    enable_bash: bool = True
    enable_note: bool = True

    # Skills
    enable_skills: bool = True
    skills_dir: str = "./skills"

    # MCP tools
    enable_mcp: bool = True
    mcp_config_path: str = "mcp.json"
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    # Tool timeouts (in seconds)
    bash_timeout: int = Field(
        default=120,
        description="Default timeout for bash tool execution (default: 120, max: 600)"
    )


class PlatformConfig(BaseModel):
    """Platform-specific configuration"""

    mode: str = Field(
        default="auto",
        description="Platform mode: 'windows', 'linux', or 'auto' (auto-detect from OS)"
    )

    @property
    def is_windows(self) -> bool:
        """Check if running on Windows platform (from config or auto-detect)"""
        if self.mode == "auto":
            import platform
            return platform.system() == "Windows"
        return self.mode.lower() == "windows"

    @property
    def is_linux(self) -> bool:
        """Check if running on Linux platform (from config or auto-detect)"""
        if self.mode == "auto":
            import platform
            return platform.system() == "Linux"
        return self.mode.lower() == "linux"

    @property
    def shell_name(self) -> str:
        """Get the appropriate shell name for current platform"""
        return "PowerShell" if self.is_windows else "bash"

    @property
    def path_separator(self) -> str:
        """Get the appropriate path separator"""
        return "\\" if self.is_windows else "/"


class M27Config(BaseModel):
    """MiniMax M2.7 specific configuration"""

    enable_extended_thinking: bool = True
    thinking_budget_tokens: int = 8192
    enable_parallel_tool_calls: bool = True
    max_concurrent_tools: int = 5
    enable_message_cache: bool = True
    token_limit: int = 800_000  # 800K tokens for 1M context window
    max_output_tokens: int = 32768  # M2.7 supports up to 32K output

    @property
    def is_m27_model(self) -> bool:
        """Check if configuration is for M2.7 model (helper method)"""
        return True  # This is M2.7 specific config class


class CLIOverrideConfig(BaseModel):
    """CLI override configuration for runtime parameter passing"""
    
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    max_steps: Optional[int] = None
    workspace_dir: Optional[str] = None
    platform_mode: Optional[str] = None
    enable_skills: Optional[bool] = None
    enable_mcp: Optional[bool] = None


class Config(BaseModel):
    """Main configuration class with environment and CLI override support"""

    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    m27: M27Config = Field(default_factory=M27Config)

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from the default search path."""
        config_path = cls.get_default_config_path()
        if not config_path.exists():
            raise FileNotFoundError("Configuration file not found. Run scripts/setup-config.sh or place config.yaml in mini_agent/config/.")
        return cls.from_yaml(config_path)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from YAML file with environment variable overrides

        Environment variables take precedence over YAML file values:
        - MINI_AGENT_API_KEY: overrides api_key
        - MINI_AGENT_API_BASE: overrides api_base
        - MINI_AGENT_MODEL: overrides model
        - MINI_AGENT_PROVIDER: overrides provider
        - MINI_AGENT_MAX_STEPS: overrides max_steps
        - MINI_AGENT_PLATFORM_MODE: overrides platform.mode

        Args:
            config_path: Configuration file path

        Returns:
            Config instance

        Raises:
            FileNotFoundError: Configuration file does not exist
            ValueError: Invalid configuration format or missing required fields
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Configuration file is empty")

        # Apply environment variable overrides first
        data = cls._apply_env_overrides(data)

        # Parse LLM configuration
        if "api_key" not in data:
            raise ValueError("Configuration file missing required field: api_key")

        if not data["api_key"] or data["api_key"] == "YOUR_API_KEY_HERE":
            raise ValueError("Please configure a valid API Key")

        api_key = data["api_key"]
        if not isinstance(api_key, str) or len(api_key) < 8:
            raise ValueError(
                f"Invalid API Key format: expected a string of at least 8 characters, "
                f"got {repr(api_key[:4])}... (length={len(api_key)})"
            )

        # Parse retry configuration
        retry_data = data.get("retry", {})
        retry_config = RetryConfig(
            enabled=retry_data.get("enabled", True),
            max_retries=retry_data.get("max_retries", 3),
            initial_delay=retry_data.get("initial_delay", 1.0),
            max_delay=retry_data.get("max_delay", 60.0),
            exponential_base=retry_data.get("exponential_base", 2.0),
        )

        llm_config = LLMConfig(
            api_key=data["api_key"],
            api_base=data.get("api_base", "https://api.minimax.io"),
            model=data.get("model", "MiniMax-M2.5"),
            provider=data.get("provider", "anthropic"),
            retry=retry_config,
        )

        # Parse Agent configuration
        agent_config = AgentConfig(
            max_steps=data.get("max_steps", 50),
            workspace_dir=data.get("workspace_dir", "./workspace"),
            system_prompt_path=data.get("system_prompt_path", "system_prompt.md"),
        )

        # Parse tools configuration
        tools_data = data.get("tools", {})

        # Parse MCP configuration
        mcp_data = tools_data.get("mcp", {})
        mcp_config = MCPConfig(
            connect_timeout=mcp_data.get("connect_timeout", 10.0),
            execute_timeout=mcp_data.get("execute_timeout", 60.0),
            sse_read_timeout=mcp_data.get("sse_read_timeout", 120.0),
        )

        tools_config = ToolsConfig(
            enable_file_tools=tools_data.get("enable_file_tools", True),
            enable_bash=tools_data.get("enable_bash", True),
            enable_note=tools_data.get("enable_note", True),
            enable_skills=tools_data.get("enable_skills", True),
            skills_dir=tools_data.get("skills_dir", "./skills"),
            enable_mcp=tools_data.get("enable_mcp", True),
            mcp_config_path=tools_data.get("mcp_config_path", "mcp.json"),
            mcp=mcp_config,
            bash_timeout=tools_data.get("bash_timeout", 120),
        )

        # Parse platform configuration
        platform_data = data.get("platform", {})
        platform_mode = platform_data.get("mode", "auto")
        platform_config = PlatformConfig(mode=platform_mode)

        # Parse M2.7 configuration
        m27_data = data.get("m27", {})
        m27_config = M27Config(
            enable_extended_thinking=m27_data.get("enable_extended_thinking", True),
            thinking_budget_tokens=m27_data.get("thinking_budget_tokens", 8192),
            enable_parallel_tool_calls=m27_data.get("enable_parallel_tool_calls", True),
            max_concurrent_tools=m27_data.get("max_concurrent_tools", 5),
            enable_message_cache=m27_data.get("enable_message_cache", True),
            token_limit=m27_data.get("token_limit", 800_000),  # 800K for 1M context
            max_output_tokens=m27_data.get("max_output_tokens", 32768),  # 32K for M2.7
        )

        return cls(
            llm=llm_config,
            agent=agent_config,
            tools=tools_config,
            platform=platform_config,
            m27=m27_config,
        )

    @classmethod
    def _apply_env_overrides(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to configuration data.

        Environment variables have the highest priority and will override
        any values in the YAML configuration file.

        Args:
            data: Configuration dictionary from YAML

        Returns:
            Updated configuration dictionary with environment overrides
        """
        # LLM configuration overrides
        if api_key := os.environ.get("MINI_AGENT_API_KEY"):
            data["api_key"] = api_key

        if api_base := os.environ.get("MINI_AGENT_API_BASE"):
            data["api_base"] = api_base

        if model := os.environ.get("MINI_AGENT_MODEL"):
            data["model"] = model

        if provider := os.environ.get("MINI_AGENT_PROVIDER"):
            data["provider"] = provider

        # Agent configuration overrides
        if max_steps := os.environ.get("MINI_AGENT_MAX_STEPS"):
            try:
                data["max_steps"] = int(max_steps)
            except ValueError:
                pass  # Keep default if invalid

        if workspace_dir := os.environ.get("MINI_AGENT_WORKSPACE_DIR"):
            data["workspace_dir"] = workspace_dir

        # Platform configuration overrides
        if platform_mode := os.environ.get("MINI_AGENT_PLATFORM_MODE"):
            if "platform" not in data:
                data["platform"] = {}
            data["platform"]["mode"] = platform_mode

        # Tools configuration overrides
        if enable_skills := os.environ.get("MINI_AGENT_ENABLE_SKILLS"):
            if "tools" not in data:
                data["tools"] = {}
            data["tools"]["enable_skills"] = enable_skills.lower() in ("true", "1", "yes")

        if enable_mcp := os.environ.get("MINI_AGENT_ENABLE_MCP"):
            if "tools" not in data:
                data["tools"] = {}
            data["tools"]["enable_mcp"] = enable_mcp.lower() in ("true", "1", "yes")

        return data

    def merge_cli_overrides(self, cli_overrides: CLIOverrideConfig) -> None:
        """Merge CLI overrides into existing configuration.

        CLI overrides have the highest priority and will override
        both YAML file values and environment variables.

        Args:
            cli_overrides: CLI override configuration object
        """
        # LLM configuration overrides
        if cli_overrides.api_key:
            self.llm.api_key = cli_overrides.api_key
        if cli_overrides.api_base:
            self.llm.api_base = cli_overrides.api_base
        if cli_overrides.model:
            self.llm.model = cli_overrides.model
        if cli_overrides.provider:
            self.llm.provider = cli_overrides.provider

        # Agent configuration overrides
        if cli_overrides.max_steps is not None:
            self.agent.max_steps = cli_overrides.max_steps
        if cli_overrides.workspace_dir:
            self.agent.workspace_dir = cli_overrides.workspace_dir

        # Platform configuration overrides
        if cli_overrides.platform_mode:
            self.platform.mode = cli_overrides.platform_mode

        # Tools configuration overrides
        if cli_overrides.enable_skills is not None:
            self.tools.enable_skills = cli_overrides.enable_skills
        if cli_overrides.enable_mcp is not None:
            self.tools.enable_mcp = cli_overrides.enable_mcp

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of the configuration
        """
        return self.model_dump()

    @staticmethod
    def get_package_dir() -> Path:
        """Get the package installation directory

        Returns:
            Path to the mini_agent package directory
        """
        # Get the directory where this config.py file is located
        return Path(__file__).parent

    @classmethod
    def find_config_file(cls, filename: str) -> Path | None:
        """Find configuration file with priority order

        Search for config file in the following order of priority:
        1) mini_agent/config/{filename} in current directory (development mode)
        2) ~/.mini-agent/config/{filename} in user home directory
        3) {package}/mini_agent/config/{filename} in package installation directory

        Args:
            filename: Configuration file name (e.g., "config.yaml", "mcp.json", "system_prompt.md")

        Returns:
            Path to found config file, or None if not found
        """
        # Priority 1: Development mode - current directory's config/ subdirectory
        dev_config = Path.cwd() / "mini_agent" / "config" / filename
        if dev_config.exists():
            return dev_config

        # Priority 2: User config directory
        user_config = Path.home() / ".mini-agent" / "config" / filename
        if user_config.exists():
            return user_config

        # Priority 3: Package installation directory's config/ subdirectory
        package_config = cls.get_package_dir() / "config" / filename
        if package_config.exists():
            return package_config

        return None

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default config file path with priority search

        Returns:
            Path to config.yaml (prioritizes: dev config/ > user config/ > package config/)
        """
        config_path = cls.find_config_file("config.yaml")
        if config_path:
            return config_path

        # Fallback to package config directory for error message purposes
        return cls.get_package_dir() / "config" / "config.yaml"

    @classmethod
    def get_env_var_help(cls) -> str:
        """Generate help text for environment variable configuration.

        Returns:
            Formatted help string for available environment variables
        """
        return """
Environment Variable Configuration:
  MINI_AGENT_API_KEY       - Override API key
  MINI_AGENT_API_BASE      - Override API base URL
  MINI_AGENT_MODEL          - Override model name
  MINI_AGENT_PROVIDER       - Override provider (anthropic/openai)
  MINI_AGENT_MAX_STEPS      - Override maximum execution steps
  MINI_AGENT_WORKSPACE_DIR  - Override workspace directory
  MINI_AGENT_PLATFORM_MODE  - Override platform mode (windows/linux/auto)
  MINI_AGENT_ENABLE_SKILLS  - Override skills enable (true/false)
  MINI_AGENT_ENABLE_MCP     - Override MCP enable (true/false)

Environment variables take precedence over config.yaml values.
"""
