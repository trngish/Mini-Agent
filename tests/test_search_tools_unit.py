from __future__ import annotations

from pathlib import Path

from mini_agent.tools.search_tools import FindTool, GrepTool


class TestGrepToolInit:
    def test_init_default_workspace(self):
        tool = GrepTool()
        assert tool.workspace_dir == Path().absolute()

    def test_init_custom_workspace(self, tmp_path):
        tool = GrepTool(str(tmp_path))
        assert tool.workspace_dir == tmp_path.absolute()


class TestGrepToolProperties:
    def test_name(self):
        tool = GrepTool()
        assert tool.name == "grep"

    def test_description_is_string(self):
        tool = GrepTool()
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_parameters_schema(self):
        tool = GrepTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "pattern" in params["properties"]
        assert "path" in params["properties"]
        assert "file_pattern" in params["properties"]
        assert "case_sensitive" in params["properties"]
        assert "regex" in params["properties"]
        assert "max_results" in params["properties"]
        assert params["required"] == ["pattern"]
        assert params["properties"]["pattern"]["type"] == "string"
        assert params["properties"]["case_sensitive"]["type"] == "boolean"
        assert params["properties"]["regex"]["type"] == "boolean"
        assert params["properties"]["max_results"]["type"] == "integer"


class TestGrepToolExecute:
    async def test_text_search(self, tmp_path):
        (tmp_path / "sample.txt").write_text("hello world\nfoo bar\nhello again", encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="hello", path=str(tmp_path))
        assert result.success is True
        assert "hello world" in result.content
        assert "hello again" in result.content

    async def test_regex_search(self, tmp_path):
        (tmp_path / "code.py").write_text("def foo():\nclass Bar:\n    pass", encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern=r"^class\s+\w+", path=str(tmp_path), regex=True)
        assert result.success is True
        assert "class Bar" in result.content

    async def test_case_sensitive(self, tmp_path):
        (tmp_path / "mixed.txt").write_text("Hello World\nhello world", encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="Hello", path=str(tmp_path), case_sensitive=True)
        assert result.success is True
        assert "Hello World" in result.content
        assert result.content.count("Hello") == 1

    async def test_case_insensitive(self, tmp_path):
        (tmp_path / "mixed.txt").write_text("Hello World\nhello world", encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="hello", path=str(tmp_path), case_sensitive=False)
        assert result.success is True
        assert "Hello World" in result.content
        assert "hello world" in result.content

    async def test_nonexistent_path(self, tmp_path):
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="hello", path=str(tmp_path / "no_such_dir"))
        assert result.success is False
        assert "Path not found" in result.error

    async def test_invalid_regex(self, tmp_path):
        (tmp_path / "dummy.txt").write_text("some text", encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="[invalid(", path=str(tmp_path), regex=True)
        assert result.success is False
        assert "Invalid regex" in result.error

    async def test_max_results_limit(self, tmp_path):
        content = "\n".join(["match line"] * 50)
        (tmp_path / "big.txt").write_text(content, encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="match", path=str(tmp_path), max_results=5)
        assert result.success is True
        assert result.content.count("match line") == 5

    async def test_no_matches(self, tmp_path):
        (tmp_path / "sample.txt").write_text("foo bar\nbaz qux", encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="not_found_text", path=str(tmp_path))
        assert result.success is True
        assert "No matches found" in result.content

    async def test_file_pattern_filter(self, tmp_path):
        (tmp_path / "code.py").write_text("searchme", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("searchme", encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        result = await tool.execute(pattern="searchme", path=str(tmp_path), file_pattern="*.py")
        assert result.success is True
        assert "code.py" in result.content
        assert "notes.txt" not in result.content


class TestGrepToolIterateFiles:
    def test_iterate_files_with_glob(self, tmp_path):
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "b.txt").write_text("", encoding="utf-8")
        (tmp_path / "c.py").write_text("", encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        files = list(tool._iterate_files(tmp_path, "*.py"))
        names = [f.name for f in files]
        assert "a.py" in names
        assert "c.py" in names
        assert "b.txt" not in names

    def test_iterate_files_single_file(self, tmp_path):
        f = tmp_path / "single.txt"
        f.write_text("content", encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        files = list(tool._iterate_files(f, "*"))
        assert len(files) == 1
        assert files[0] == f

    def test_iterate_files_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden_dir"
        hidden.mkdir()
        (hidden / "secret.txt").write_text("secret", encoding="utf-8")
        (tmp_path / "visible.txt").write_text("visible", encoding="utf-8")
        tool = GrepTool(str(tmp_path))
        files = list(tool._iterate_files(tmp_path, "*"))
        names = [f.name for f in files]
        assert "visible.txt" in names
        assert "secret.txt" not in names


class TestFindToolInit:
    def test_init_default_workspace(self):
        tool = FindTool()
        assert tool.workspace_dir == Path().absolute()

    def test_init_custom_workspace(self, tmp_path):
        tool = FindTool(str(tmp_path))
        assert tool.workspace_dir == tmp_path.absolute()


class TestFindToolProperties:
    def test_name(self):
        tool = FindTool()
        assert tool.name == "find"

    def test_description_is_string(self):
        tool = FindTool()
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_parameters_schema(self):
        tool = FindTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "pattern" in params["properties"]
        assert "path" in params["properties"]
        assert "case_sensitive" in params["properties"]
        assert "max_results" in params["properties"]
        assert params["required"] == ["pattern"]
        assert params["properties"]["pattern"]["type"] == "string"
        assert params["properties"]["case_sensitive"]["type"] == "boolean"
        assert params["properties"]["max_results"]["type"] == "integer"


class TestFindToolExecute:
    async def test_find_by_pattern(self, tmp_path):
        (tmp_path / "app.py").write_text("", encoding="utf-8")
        (tmp_path / "readme.md").write_text("", encoding="utf-8")
        (tmp_path / "utils.py").write_text("", encoding="utf-8")
        tool = FindTool(str(tmp_path))
        result = await tool.execute(pattern="*.py", path=str(tmp_path))
        assert result.success is True
        assert "app.py" in result.content
        assert "utils.py" in result.content
        assert "readme.md" not in result.content

    async def test_find_nonexistent_path(self, tmp_path):
        tool = FindTool(str(tmp_path))
        result = await tool.execute(pattern="*.py", path=str(tmp_path / "no_such_dir"))
        assert result.success is False
        assert "Path not found" in result.error

    async def test_find_case_insensitive(self, tmp_path):
        (tmp_path / "README.MD").write_text("", encoding="utf-8")
        tool = FindTool(str(tmp_path))
        result = await tool.execute(pattern="*.md", path=str(tmp_path), case_sensitive=False)
        assert result.success is True
        assert "README.MD" in result.content

    async def test_find_case_sensitive(self, tmp_path):
        (tmp_path / "readme.md").write_text("", encoding="utf-8")
        (tmp_path / "DATA.TXT").write_text("", encoding="utf-8")
        tool = FindTool(str(tmp_path))
        result = await tool.execute(pattern="*.md", path=str(tmp_path), case_sensitive=True)
        assert result.success is True
        assert "readme.md" in result.content
        assert "DATA.TXT" not in result.content

    async def test_find_max_results(self, tmp_path):
        for i in range(20):
            (tmp_path / f"file_{i:02d}.py").write_text("", encoding="utf-8")
        tool = FindTool(str(tmp_path))
        result = await tool.execute(pattern="*.py", path=str(tmp_path), max_results=5)
        assert result.success is True
        assert "showing first 5" in result.content

    async def test_find_no_matches(self, tmp_path):
        (tmp_path / "app.py").write_text("", encoding="utf-8")
        tool = FindTool(str(tmp_path))
        result = await tool.execute(pattern="*.java", path=str(tmp_path))
        assert result.success is True
        assert "No files found" in result.content

    async def test_find_nested_directories(self, tmp_path):
        sub = tmp_path / "src" / "deep"
        sub.mkdir(parents=True)
        (sub / "nested.py").write_text("", encoding="utf-8")
        tool = FindTool(str(tmp_path))
        result = await tool.execute(pattern="*.py", path=str(tmp_path))
        assert result.success is True
        assert "nested.py" in result.content
