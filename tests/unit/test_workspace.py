"""Unit tests for workspace module."""
import pytest
from pathlib import Path
from tools.workspace import WorkspaceManager


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace manager with temp directory."""
    wm = WorkspaceManager()
    wm.configure(str(tmp_path))
    yield wm
    # Cleanup
    wm._initialized = False
    wm._output_path = None
    wm._project_root = None
    wm._project_name = None


def test_configure_and_new_project(workspace, tmp_path):
    """Test project creation."""
    project_root = workspace.new_project("test project")
    
    assert workspace.is_initialized
    assert workspace.project_name == "test project"
    assert workspace.project_root == project_root
    assert project_root.exists()
    
    # Check subdirectories created
    for subdir in ["ml_engineer", "data_scientist", "data_analyst", "github", "shared"]:
        assert (project_root / subdir).exists()


def test_write_and_read(workspace, tmp_path):
    """Test file write and read."""
    workspace.new_project("test")
    
    content = "test content"
    path = workspace.write("data_scientist", "test.txt", content)
    
    assert path.exists()
    assert path.read_text() == content
    
    read_content = workspace.read("data_scientist", "test.txt")
    assert read_content == content


def test_write_bytes(workspace, tmp_path):
    """Test binary file write."""
    workspace.new_project("test")
    
    data = b"binary content"
    path = workspace.write_bytes("shared", "data.bin", data)
    
    assert path.exists()
    assert path.read_bytes() == data


def test_list_files(workspace, tmp_path):
    """Test file listing."""
    workspace.new_project("test")
    
    workspace.write("ml_engineer", "model.py", "code")
    workspace.write("data_analyst", "report.md", "report")
    workspace.write_bytes("shared", "data.csv", b"csv")
    
    files = workspace.list_files()
    assert len(files) == 3
    # Use platform-agnostic path separators
    assert any("ml_engineer" in f and "model.py" in f for f in files)
    assert any("data_analyst" in f and "report.md" in f for f in files)
    assert any("shared" in f and "data.csv" in f for f in files)


def test_list_files_by_agent(workspace, tmp_path):
    """Test file listing filtered by agent."""
    workspace.new_project("test")
    
    workspace.write("ml_engineer", "a.py", "a")
    workspace.write("ml_engineer", "b.py", "b")
    workspace.write("data_scientist", "c.py", "c")
    
    ml_files = workspace.list_files("ml_engineer")
    assert len(ml_files) == 2
    assert all("ml_engineer" in f for f in ml_files)


def test_copy_to_shared(workspace, tmp_path):
    """Test copying file to shared."""
    workspace.new_project("test")
    
    workspace.write("data_scientist", "features.csv", "feature data")
    shared_path = workspace.copy_to_shared("data_scientist", "features.csv")
    
    assert shared_path.exists()
    assert shared_path.read_text() == "feature data"
    assert shared_path == workspace.project_root / "shared" / "features.csv"


def test_load_existing_project(workspace, tmp_path):
    """Test loading an existing project."""
    # Create project
    project_root = workspace.new_project("original")
    workspace.write("shared", "data.csv", "csv data")
    
    # Create new workspace manager and load
    wm2 = WorkspaceManager()
    wm2.configure(str(tmp_path))
    loaded_root = wm2.load_project(project_root)
    
    assert wm2.is_initialized
    assert wm2.project_root == project_root
    assert wm2.project_name == "original"
    assert wm2.read("shared", "data.csv") == "csv data"