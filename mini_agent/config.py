"""配置管理模块

提供统一的配置加载和管理功能，支持:
- YAML配置文件
- 环境变量覆盖
- CLI参数合并
- 配置验证
"""

import contextlib
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_serializer


class RetryConfig(BaseModel):
    """YAML配置文件的重试配置"""

    enabled: bool = True
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class LLMConfig(BaseModel):
    """LLM配置"""

    api_key: str
    api_base: str = "https://api.minimax.io"
    model: str = "MiniMax-M2.5"
    provider: str = "anthropic"  # 可选值: "anthropic" 或 "openai"
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        """Serialize model with api_key masked for安全."""
        data = handler(self)
        if data.get("api_key"):
            # Mask API key: show first 4 and last 4 chars
            key = data["api_key"]
            if len(key) > 8:
                data["api_key"] = f"{key[:4]}***{key[-4:]}"
            else:
                data["api_key"] = "****"
        return data

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or v == "YOUR_API_KEY_HERE":
            raise ValueError(
                "API密钥未配置。请设置 MINIMAX_API_KEY 环境变量或在 config.yaml 中配置 api_key"
            )
        if len(v) < 8:
            raise ValueError(
                f"API密钥格式无效: 期望至少8个字符，实际得到 {repr(v[:4])}... (长度={len(v)})"
            )
        return v


class AgentConfig(BaseModel):
    """Agent配置"""

    max_steps: int = 50
    workspace_dir: str = "./workspace"
    system_prompt_path: str = "system_prompt.md"


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) 超时配置"""

    connect_timeout: float = 10.0  # 连接超时时间（秒）
    execute_timeout: float = 60.0  # 工具执行超时时间（秒）
    sse_read_timeout: float = 120.0  # SSE读取超时时间（秒）


class ToolsConfig(BaseModel):
    """ "工具配置"""

    # 基础工具（文件操作、bash）
    enable_file_tools: bool = True
    enable_bash: bool = True
    enable_note: bool = True

    # 技能（Skills）
    enable_skills: bool = True
    skills_dir: str = "./skills"

    # MCP工具
    enable_mcp: bool = True
    mcp_config_path: str = "mcp.json"
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    # 工具超时时间（秒）
    bash_timeout: int = Field(
        default=120, description="bash工具执行的默认超时时间（默认: 120, 最大: 600）"
    )

    def get_skills_search_paths(self) -> list[Path]:
        """ "获取技能目录搜索列表，按优先级排序。

        优先级:
        1. 用户配置目录: ~/.mini-agent/skills/
        2. 项目目录: {project_root}/mini_agent/skills/

        Returns:
            技能目录路径列表
        """
        paths = []

        # 优先级1: 用户配置目录的技能 (~/.mini-agent/skills/)
        user_skills_dir = Path.home() / ".mini-agent" / "skills"
        if user_skills_dir.exists():
            paths.append(user_skills_dir)

        # 优先级2: 项目目录的技能 (./mini_agent/skills/)
        project_skills_dir = Path("mini_agent") / "skills"
        if project_skills_dir.exists():
            paths.append(project_skills_dir)

        return paths

    def get_mcp_config_paths(self) -> list[Path]:
        """获取MCP配置文件搜索路径，按优先级排序。

        优先级:
        1. 用户配置目录: ~/.mini-agent/config/mcp.json
        2. 项目目录: ./mcp.json

        Returns:
            MCP配置文件路径列表
        """
        paths = []

        # 优先级1: 用户配置目录的MCP配置 (~/.mini-agent/config/mcp.json)
        user_mcp_config = Path.home() / ".mini-agent" / "config" / "mcp.json"
        if user_mcp_config.exists():
            paths.append(user_mcp_config)

        # 优先级2: 项目目录的MCP配置 (./mcp.json)
        project_mcp_config = Path("mcp.json")
        if project_mcp_config.exists():
            paths.append(project_mcp_config)

        return paths


class PlatformConfig(BaseModel):
    """平台相关配置"""

    mode: str = Field(default="auto", description="平台模式: 'windows', 'linux', 或 'auto'（自动检测操作系统）")


class SecurityConfig(BaseModel):
    """路径访问控制的安全配置"""

    extra_blocked_dirs: list[str] = Field(
        default_factory=list,
        description="要阻止的额外目录（除内置系统目录外）",
    )
    extra_blocked_home_subdirs: list[str] = Field(
        default_factory=list,
        description="要阻止的额外主目录子目录（除 .ssh, .gnupg, .config/ssh 外）",
    )


class M27Config(BaseModel):
    """MiniMax M2.7特定配置"""

    enable_extended_thinking: bool = True
    thinking_budget_tokens: int = 32768  # 按调用计费: 完整32K预算，越深越准确=调用次数越少
    thinking_budget_adaptive: bool = True  # 自适应思考预算
    enable_message_cache: bool = True
    enable_parallel_tool_calls: bool = True
    max_concurrent_tools: int = 20  # M2.7支持20+并行调用，单次调用越多=调用次数越少
    token_limit: int = 800_000  # 800K tokens对应1M上下文窗口
    max_output_tokens: int = 32768  # M2.7支持最多32K输出


class CLIOverrideConfig(BaseModel):
    """运行时参数传递的CLI覆盖配置"""

    api_key: str | None = None
    api_base: str | None = None
    model: str | None = None
    provider: str | None = None
    max_steps: int | None = None
    workspace_dir: str | None = None
    platform_mode: str | None = None
    enable_skills: bool | None = None
    enable_mcp: bool | None = None


class Config(BaseModel):
    """支持环境和CLI覆盖的主配置类"""

    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    m27: M27Config = Field(default_factory=M27Config)

    @classmethod
    def load(cls) -> "Config":
        """从默认搜索路径加载配置。"""
        config_path = cls.get_default_config_path()
        if not config_path.exists():
            raise FileNotFoundError(
                "未找到配置文件。请运行 scripts/setup-config.sh 或将 config.yaml 放置在 mini_agent/config/ 目录中。"
            )
        return cls.from_yaml(config_path)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """从YAML文件加载配置，并应用环境变量覆盖

        环境变量优先于YAML文件中的值:
        - MINI_AGENT_API_KEY: 覆盖 api_key
        - MINI_AGENT_API_BASE: 覆盖 api_base
        - MINI_AGENT_MODEL: 覆盖 model
        - MINI_AGENT_PROVIDER: 覆盖 provider
        - MINI_AGENT_MAX_STEPS: 覆盖 max_steps
        - MINI_AGENT_PLATFORM_MODE: 覆盖 platform.mode

        Args:
            config_path: 配置文件路径

        Returns:
            Config实例

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置格式无效或缺少必需字段
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("配置文件为空")

        # 首先应用环境变量覆盖
        data = cls._apply_env_overrides(data)

        # 解析所有配置节
        config = cls._parse_config(data, config_path)

        # 使用ConfigValidator验证配置
        from .utils.config_validator import ConfigValidator

        ConfigValidator.validate_or_raise(config)

        return config

    @classmethod
    def _parse_config(cls, data: dict[str, Any], config_path: Path) -> "Config":
        """从数据字典解析所有配置节。

        Args:
            data: 来自YAML的配置字典
            config_path: 用于解析相对路径的原始配置文件路径

        Returns:
            Config实例
        """
        llm_config = cls._parse_llm_config(data)
        agent_config = cls._parse_agent_config(data)
        tools_config = cls._parse_tools_config(data, config_path)
        platform_config = cls._parse_platform_config(data)
        security_config = cls._parse_security_config(data)
        m27_config = cls._parse_m27_config(data)

        return cls(
            llm=llm_config,
            agent=agent_config,
            tools=tools_config,
            platform=platform_config,
            security=security_config,
            m27=m27_config,
        )

    @classmethod
    def _parse_llm_config(cls, data: dict[str, Any]) -> LLMConfig:
        """解析LLM配置节。"""
        retry_data = data.get("retry", {})
        retry_config = RetryConfig(
            enabled=retry_data.get("enabled", True),
            max_retries=retry_data.get("max_retries", 3),
            initial_delay=retry_data.get("initial_delay", 1.0),
            max_delay=retry_data.get("max_delay", 60.0),
            exponential_base=retry_data.get("exponential_base", 2.0),
        )

        api_key = data.get("api_key")
        if not api_key:
            raise ValueError(
                "api_key 是必需的但未找到。请设置 MINIMAX_API_KEY 环境变量或在 config.yaml 中添加 api_key"
            )

        return LLMConfig(
            api_key=api_key,
            api_base=data.get("api_base", "https://api.minimax.io"),
            model=data.get("model", "MiniMax-M2.5"),
            provider=data.get("provider", "anthropic"),
            retry=retry_config,
        )

    @classmethod
    def _parse_agent_config(cls, data: dict[str, Any]) -> AgentConfig:
        """解析Agent配置节。"""
        return AgentConfig(
            max_steps=data.get("max_steps", AgentConfig().max_steps),
            workspace_dir=data.get("workspace_dir", "./workspace"),
            system_prompt_path=data.get("system_prompt_path", "system_prompt.md"),
        )

    @classmethod
    def _parse_tools_config(cls, data: dict[str, Any], config_path: Path) -> ToolsConfig:
        """解析Tools配置节。"""
        tools_data = data.get("tools", {})

        # 解析MCP配置
        mcp_data = tools_data.get("mcp", {})
        mcp_config = MCPConfig(
            connect_timeout=mcp_data.get("connect_timeout", 10.0),
            execute_timeout=mcp_data.get("execute_timeout", 60.0),
            sse_read_timeout=mcp_data.get("sse_read_timeout", 120.0),
        )

        # 相对于项目根目录解析skills_dir（配置文件位于 mini_agent/config/）
        skills_dir_raw = tools_data.get("skills_dir", "./mini_agent/skills")
        skills_dir_path = Path(skills_dir_raw)
        if not skills_dir_path.is_absolute():
            project_root = config_path.parent.parent
            skills_dir_path = (project_root / skills_dir_path).resolve()

        return ToolsConfig(
            enable_file_tools=tools_data.get("enable_file_tools", True),
            enable_bash=tools_data.get("enable_bash", True),
            enable_note=tools_data.get("enable_note", True),
            enable_skills=tools_data.get("enable_skills", True),
            skills_dir=str(skills_dir_path),
            enable_mcp=tools_data.get("enable_mcp", True),
            mcp_config_path=tools_data.get("mcp_config_path", "mcp.json"),
            mcp=mcp_config,
            bash_timeout=tools_data.get("bash_timeout", 120),
        )

    @classmethod
    def _parse_platform_config(cls, data: dict[str, Any]) -> PlatformConfig:
        """解析Platform配置节。"""
        platform_data = data.get("platform", {})
        platform_mode = platform_data.get("mode", "auto")
        return PlatformConfig(mode=platform_mode)

    @classmethod
    def _parse_security_config(cls, data: dict[str, Any]) -> SecurityConfig:
        """解析Security配置节。"""
        security_data = data.get("security", {})
        return SecurityConfig(
            extra_blocked_dirs=security_data.get("extra_blocked_dirs", []),
            extra_blocked_home_subdirs=security_data.get("extra_blocked_home_subdirs", []),
        )

    @classmethod
    def _parse_m27_config(cls, data: dict[str, Any]) -> M27Config:
        """解析M2.7配置节。"""
        m27_data = data.get("m27", {})
        return M27Config(
            enable_extended_thinking=m27_data.get("enable_extended_thinking", True),
            thinking_budget_tokens=m27_data.get("thinking_budget_tokens", 32768),
            thinking_budget_adaptive=m27_data.get("thinking_budget_adaptive", True),
            enable_parallel_tool_calls=m27_data.get("enable_parallel_tool_calls", True),
            max_concurrent_tools=m27_data.get("max_concurrent_tools", 20),
            token_limit=m27_data.get("token_limit", 800_000),
            max_output_tokens=m27_data.get("max_output_tokens", 32768),
        )

    @classmethod
    def _apply_env_overrides(cls, data: dict[str, Any]) -> dict[str, Any]:
        """应用环境变量覆盖到配置数据。

        安全注意: 环境变量具有最高优先级，
        会覆盖YAML配置文件中的任何值。

        支持的环境变量:
        - MINIMAX_API_KEY: API密钥（必需，最少8个字符）
        - MINI_AGENT_API_KEY: API密钥（兼容性别名）
        - MINI_AGENT_API_BASE: API基础URL
        - MINI_AGENT_MODEL: 模型名称
        - MINI_AGENT_PROVIDER: 提供商类型
        - MINI_AGENT_MAX_STEPS: 最大执行步骤
        - MINI_AGENT_WORKSPACE_DIR: 工作目录
        - MINI_AGENT_PLATFORM_MODE: 平台模式（windows/linux/auto）

        对于生产部署，建议:
        1. 仅通过环境变量设置API_KEY
        2. 生产环境使用只读配置文件
        3. 通过ConfigValidator验证所有覆盖

        Args:
            data: 来自YAML的配置字典

        Returns:
            更新后的配置字典（包含环境覆盖）
        """
        # LLM配置覆盖
        # 同时支持 MINIMAX_API_KEY（常规）和 MINI_AGENT_API_KEY（兼容）
        if api_key := (os.environ.get("MINIMAX_API_KEY") or os.environ.get("MINI_AGENT_API_KEY")):
            data["api_key"] = api_key

        if api_base := os.environ.get("MINI_AGENT_API_BASE"):
            data["api_base"] = api_base

        if model := os.environ.get("MINI_AGENT_MODEL"):
            data["model"] = model

        if provider := os.environ.get("MINI_AGENT_PROVIDER"):
            data["provider"] = provider

        # Agent配置覆盖
        if max_steps := os.environ.get("MINI_AGENT_MAX_STEPS"):
            with contextlib.suppress(ValueError):
                data["max_steps"] = int(max_steps)

        if workspace_dir := os.environ.get("MINI_AGENT_WORKSPACE_DIR"):
            data["workspace_dir"] = workspace_dir

        # Platform配置覆盖
        if platform_mode := os.environ.get("MINI_AGENT_PLATFORM_MODE"):
            if "platform" not in data:
                data["platform"] = {}
            data["platform"]["mode"] = platform_mode

        # Tools配置覆盖
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
        """合并CLI覆盖到现有配置。

        CLI覆盖具有最高优先级，会覆盖
        YAML文件值和环境变量。

        Args:
            cli_overrides: CLI覆盖配置对象
        """
        # LLM配置覆盖
        if cli_overrides.api_key:
            self.llm.api_key = cli_overrides.api_key
        if cli_overrides.api_base:
            self.llm.api_base = cli_overrides.api_base
        if cli_overrides.model:
            self.llm.model = cli_overrides.model
        if cli_overrides.provider:
            self.llm.provider = cli_overrides.provider

        # Agent配置覆盖
        if cli_overrides.max_steps is not None:
            self.agent.max_steps = cli_overrides.max_steps
        if cli_overrides.workspace_dir:
            self.agent.workspace_dir = cli_overrides.workspace_dir

        # Platform配置覆盖
        if cli_overrides.platform_mode:
            self.platform.mode = cli_overrides.platform_mode

        # Tools配置覆盖
        if cli_overrides.enable_skills is not None:
            self.tools.enable_skills = cli_overrides.enable_skills
        if cli_overrides.enable_mcp is not None:
            self.tools.enable_mcp = cli_overrides.enable_mcp

        from .utils.config_validator import ConfigValidator

        ConfigValidator.validate_or_raise(self)

    def to_dict(self) -> dict[str, Any]:
        """将配置转换为字典用于序列化。

        Returns:
            配置的字典表示
        """
        return self.model_dump()

    @staticmethod
    def get_package_dir() -> Path:
        """获取包安装目录

        Returns:
            mini_agent包目录的路径
        """
        # 获取config.py文件所在目录
        return Path(__file__).parent

    @classmethod
    def find_config_file(cls, filename: str) -> Path | None:
        """按优先级顺序查找配置文件

        按以下优先级搜索配置文件:
        1) 当前目录的 mini_agent/config/{filename}（开发模式）
        2) 用户主目录的 ~/.mini-agent/config/{filename}
        3) 包安装目录的 {package}/mini_agent/config/{filename}

        Args:
            filename: 配置文件名（例如 "config.yaml", "mcp.json", "system_prompt.md"）

        Returns:
            找到的配置文件的路径，若未找到则返回None
        """
        # 优先级1: 开发模式 - 当前目录的config/子目录
        dev_config = Path.cwd() / "mini_agent" / "config" / filename
        if dev_config.exists():
            return dev_config

        # 优先级2: 用户配置目录
        user_config = Path.home() / ".mini-agent" / "config" / filename
        if user_config.exists():
            return user_config

        # 优先级3: 包安装目录的config/子目录
        package_config = cls.get_package_dir() / "config" / filename
        if package_config.exists():
            return package_config

        return None

    @classmethod
    def get_default_config_path(cls) -> Path:
        """获取默认配置文件路径，按优先级搜索

        Returns:
            config.yaml的路径（优先级: 开发配置/ > 用户配置/ > 包配置/）
        """
        config_path = cls.find_config_file("config.yaml")
        if config_path:
            return config_path

        # 为错误信息提供后备的包配置目录
        return cls.get_package_dir() / "config" / "config.yaml"

    @classmethod
    def get_env_var_help(cls) -> str:
        """生成环境变量配置的帮助文本。

        Returns:
            可用环境变量的格式化帮助字符串
        """
        return """
环境变量配置:
  MINIMAX_API_KEY          - 覆盖API密钥（推荐）
  MINI_AGENT_API_KEY       - 覆盖API密钥（兼容性别名）
  MINI_AGENT_API_BASE      - 覆盖API基础URL
  MINI_AGENT_MODEL         - 覆盖模型名称
  MINI_AGENT_PROVIDER      - 覆盖提供商（anthropic/openai）
  MINI_AGENT_MAX_STEPS     - 覆盖最大执行步骤
  MINI_AGENT_WORKSPACE_DIR - 覆盖工作目录
  MINI_AGENT_PLATFORM_MODE - 覆盖平台模式（windows/linux/auto）
  MINI_AGENT_ENABLE_SKILLS - 覆盖技能启用（true/false）
  MINI_AGENT_ENABLE_MCP    - 覆盖MCP启用（true/false）

环境变量优先于config.yaml中的值。
MINIMAX_API_KEY是推荐的变量；MINI_AGENT_API_KEY是兼容性别名。
"""
