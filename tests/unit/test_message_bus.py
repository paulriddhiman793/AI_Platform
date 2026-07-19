"""Unit tests for message bus."""
import asyncio
import pytest
from api.message_bus import InMemoryMessageBus, _dedup_p2p


@pytest.fixture
def bus():
    """Create a fresh message bus for each test."""
    return InMemoryMessageBus()


@pytest.mark.asyncio
async def test_subscribe_and_publish(bus):
    """Test basic subscribe and publish."""
    channel = "test.channel"
    queue = bus.subscribe(channel)
    
    message = {"from": "agent1", "to": "agent2", "content": "hello"}
    await bus.publish(channel, message)
    
    envelope = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert envelope["channel"] == channel
    assert envelope["payload"] == message


@pytest.mark.asyncio
async def test_multiple_subscribers(bus):
    """Test multiple subscribers receive messages."""
    channel = "test.multi"
    q1 = bus.subscribe(channel)
    q2 = bus.subscribe(channel)
    
    message = {"content": "broadcast"}
    await bus.publish(channel, message)
    
    env1 = await asyncio.wait_for(q1.get(), timeout=1.0)
    env2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    
    assert env1["payload"] == message
    assert env2["payload"] == message


@pytest.mark.asyncio
async def test_subscriber_count(bus):
    """Test subscriber count tracking."""
    channel = "test.count"
    assert bus.subscriber_count(channel) == 0
    
    q1 = bus.subscribe(channel)
    assert bus.subscriber_count(channel) == 1
    
    q2 = bus.subscribe(channel)
    assert bus.subscriber_count(channel) == 2


def test_dedup_p2p():
    """Test P2P deduplication."""
    # First call should not be duplicate
    assert not _dedup_p2p("agent1", "agent2", "hello")
    # Second call within window should be duplicate
    assert _dedup_p2p("agent1", "agent2", "hello")
    # Different content should not be duplicate
    assert not _dedup_p2p("agent1", "agent2", "world")


def test_dedup_p2p_window():
    """Test deduplication window expiration."""
    import time
    from api.message_bus import _recent_p2p
    
    _recent_p2p.clear()
    # Add an old entry
    _recent_p2p[("agent1", "agent2", "old")] = time.time() - 10
    
    # Should not be duplicate (outside window)
    assert not _dedup_p2p("agent1", "agent2", "old")
    
    # Cleanup
    _recent_p2p.clear()