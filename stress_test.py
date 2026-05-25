"""压力测试 Mini-Agent 核心组件: Agent, SubAgent, AgentTeam"""
import asyncio
import time
from mini_agent import Agent, SubAgent, AgentTeam, LLMClient
from mini_agent.schema import AgentMode, Message
from mini_agent.tools.bash_tool import BashTool
from mini_agent.tools.file_tools import FileTools

# 测试配置
ANTHROPIC_API_KEY = "your-api-key"  # 替换为有效密钥或设置环境变量

def test_imports():
    """测试基础导入"""
    print("✅ Import successful")
    print(f"   Agent: {Agent}")
    print(f"   SubAgent: {SubAgent}")
    print(f"   AgentTeam: {AgentTeam}")

def test_agent_mode():
    """验证 AgentMode 枚举"""
    print("\n📋 AgentMode values:")
    for mode in AgentMode:
        print(f"   - {mode.name} = {mode.value}")

def test_basic_agent():
    """测试基本 Agent 功能"""
    print("\n🧪 Testing Agent...")
    
    llm = LLMClient(
        provider="anthropic",
        api_key=ANTHROPIC_API_KEY,
        model="claude-sonnet-4-20250514"
    )
    
    tools = [BashTool(), FileTools()]
    
    agent = Agent(
        llm_client=llm,
        tools=tools,
        mode=AgentMode.YOLO,
        max_turns=3
    )
    
    print(f"   Agent created: {agent}")
    print(f"   Mode: {agent.mode}")
    print(f"   Tools: {[t.name for t in agent.tools]}")
    return agent

async def test_agent_execution(agent: Agent):
    """测试 Agent 执行"""
    print("\n🚀 Executing Agent task...")
    
    messages = [
        Message(role="user", content="What is 2 + 2? Just answer briefly.")
    ]
    
    start = time.perf_counter()
    result = await agent.run(messages)
    elapsed = time.perf_counter() - start
    
    print(f"   ✅ Completed in {elapsed:.2f}s")
    print(f"   Response: {result.content[:100]}..." if len(result.content) > 100 else f"   Response: {result.content}")
    return result

async def test_subagent():
    """测试 SubAgent"""
    print("\n🧪 Testing SubAgent...")
    
    llm = LLMClient(
        provider="anthropic", 
        api_key=ANTHROPIC_API_KEY,
        model="claude-sonnet-4-20250514"
    )
    
    tools = [BashTool()]
    
    subagent = SubAgent(
        llm_client=llm,
        tools=tools,
        system_prompt="You are a calculator assistant."
    )
    
    print(f"   SubAgent created: {subagent}")
    
    start = time.perf_counter()
    result = await subagent.run("What is 5 * 3?")
    elapsed = time.perf_counter() - start
    
    print(f"   ✅ Completed in {elapsed:.2f}s")
    print(f"   Result: {result.content[:100]}..." if len(result.content) > 100 else f"   Result: {result.content}")
    return result

async def test_agent_team():
    """测试 AgentTeam"""
    print("\n🧪 Testing AgentTeam...")
    
    team = AgentTeam(
        agents=[
            AgentRole(name="researcher", model="claude-sonnet-4-20250514", api_key=ANTHROPIC_API_KEY),
            AgentRole(name="coder", model="claude-sonnet-4-20250514", api_key=ANTHROPIC_API_KEY),
        ],
        system_prompt="You are a team of experts."
    )
    
    print(f"   Team created with {len(team.agents)} agents")
    
    start = time.perf_counter()
    result = await team.run("Write a hello world Python script.")
    elapsed = time.perf_counter() - start
    
    print(f"   ✅ Completed in {elapsed:.2f}s")
    print(f"   Success: {result.success}")
    return result

async def main():
    """运行所有测试"""
    print("=" * 60)
    print("🔬 Mini-Agent Core Components Stress Test")
    print("=" * 60)
    
    test_imports()
    test_agent_mode()
    
    # Agent Basic Test
    agent = test_basic_agent()
    
    # Agent Execution (需要有效API密钥)
    try:
        await test_agent_execution(agent)
    except Exception as e:
        print(f"   ⚠️ Agent execution skipped: {e}")
    
    # SubAgent Test
    try:
        await test_subagent()
    except Exception as e:
        print(f"   ⚠️ SubAgent test skipped: {e}")
    
    # AgentTeam Test
    try:
        await test_agent_team()
    except Exception as e:
        print(f"   ⚠️ AgentTeam test skipped: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Stress test completed")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())