"""
GEPA Monitoring Dashboard.

A simple web UI to monitor GEPA optimization status.
"""

from datetime import datetime, timedelta
from pathlib import Path
import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import structlog

logger = structlog.get_logger()

app = FastAPI(title="GEPA Dashboard")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard view."""
    from src.core.gepa.automation import AutoGEPAConfig, StateManager
    from src.core.gepa import OptimizationStore
    
    config = AutoGEPAConfig.from_env()
    state_manager = StateManager(config)
    store = OptimizationStore("./data/gepa")
    
    state = state_manager.state
    
    # Get recent traces
    traces = store.load_traces()
    recent_traces = sorted(traces, key=lambda t: t.started_at, reverse=True)[:20]
    
    # Get recent escalations
    escalations = []
    if config.escalation_log.exists():
        with open(config.escalation_log) as f:
            content = f.read()
            # Parse escalations (simple approach)
            for block in content.split("="*60):
                if "GEPA AUTOMATION ALERT" in block:
                    escalations.append(block.strip()[:500])
    
    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GEPA Dashboard</title>
        <meta http-equiv="refresh" content="60">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .card {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .status-ok {{ color: #28a745; }}
            .status-warning {{ color: #ffc107; }}
            .status-error {{ color: #dc3545; }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{ background: #f8f9fa; }}
            .metric {{
                display: inline-block;
                padding: 20px;
                text-align: center;
                min-width: 150px;
            }}
            .metric-value {{
                font-size: 36px;
                font-weight: bold;
            }}
            .metric-label {{
                color: #666;
                font-size: 14px;
            }}
            pre {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <h1>🤖 GEPA Optimization Dashboard</h1>
        <p style="color: #666;">Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        
        <div class="card">
            <h2>System Status</h2>
            <div class="metric">
                <div class="metric-value status-ok">●</div>
                <div class="metric-label">Automation Loop</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(traces)}</div>
                <div class="metric-label">Total Traces</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(state.last_optimization)}</div>
                <div class="metric-label">Optimized Agents</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(escalations)}</div>
                <div class="metric-label">Escalations</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Agent Performance</h2>
            <table>
                <tr>
                    <th>Agent Type</th>
                    <th>Pending Traces</th>
                    <th>Current Score</th>
                    <th>Baseline Score</th>
                    <th>Last Optimization</th>
                    <th>Status</th>
                </tr>
    """
    
    for agent_type in set(state.trace_counts.keys()) | set(state.current_scores.keys()):
        trace_count = state.trace_counts.get(agent_type, 0)
        current = state.current_scores.get(agent_type, 0)
        baseline = state.baseline_scores.get(agent_type, current)
        last_opt = state.last_optimization.get(agent_type, "Never")
        
        # Determine status
        should_opt, reason = state_manager.should_optimize(agent_type)
        if should_opt:
            status = f'<span class="status-warning">⚠️ {reason}</span>'
        elif current >= baseline * 0.95:
            status = '<span class="status-ok">✓ Healthy</span>'
        else:
            status = '<span class="status-warning">⚠️ Below baseline</span>'
        
        diff = current - baseline
        diff_str = f"{diff:+.3f}" if baseline > 0 else "N/A"
        
        html += f"""
                <tr>
                    <td><strong>{agent_type}</strong></td>
                    <td>{trace_count}</td>
                    <td>{current:.3f}</td>
                    <td>{baseline:.3f} ({diff_str})</td>
                    <td>{last_opt}</td>
                    <td>{status}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="card">
            <h2>Recent Traces</h2>
            <table>
                <tr>
                    <th>Time</th>
                    <th>Agent</th>
                    <th>Task</th>
                    <th>Result</th>
                    <th>Duration</th>
                </tr>
    """
    
    for trace in recent_traces[:10]:
        status = '✓' if trace.success else '✗'
        status_class = 'status-ok' if trace.success else 'status-error'
        task_preview = trace.instructions[:50] + "..." if len(trace.instructions) > 50 else trace.instructions
        
        html += f"""
                <tr>
                    <td>{trace.started_at.strftime('%H:%M:%S')}</td>
                    <td>{trace.agent_type}</td>
                    <td>{task_preview}</td>
                    <td class="{status_class}">{status}</td>
                    <td>{trace.duration_seconds:.1f}s</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    """
    
    if escalations:
        html += """
        <div class="card">
            <h2>Recent Escalations</h2>
        """
        for esc in escalations[-3:]:
            html += f"<pre>{esc[:1000]}</pre>"
        html += "</div>"
    
    html += """
        <div class="card">
            <h2>Quick Actions</h2>
            <p>
                <a href="/trigger-optimization/browser">🔄 Trigger Browser Agent Optimization</a> |
                <a href="/status">📊 JSON Status</a> |
                <a href="/diagnose">🔍 Run Diagnostics</a>
            </p>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)


@app.get("/status")
async def status():
    """JSON status endpoint."""
    from src.core.gepa.automation import AutoGEPAConfig, StateManager
    
    config = AutoGEPAConfig.from_env()
    state_manager = StateManager(config)
    
    return {
        "status": "running",
        "state": state_manager.state.to_dict(),
        "config": {
            "min_traces": config.min_traces_for_optimization,
            "interval_hours": config.optimization_interval_hours,
            "budget": config.default_budget,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/trigger-optimization/{agent_type}")
async def trigger_optimization(agent_type: str):
    """Manually trigger optimization for an agent."""
    from src.core.gepa.auto_loop import GEPAAutomationLoop
    
    loop = GEPAAutomationLoop()
    
    # Run optimization
    try:
        await loop._run_optimization(agent_type, "Manual trigger")
        return {"status": "success", "message": f"Optimization triggered for {agent_type}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/diagnose")
async def diagnose():
    """Run diagnostics."""
    from src.core.gepa.auto_loop import GEPAAutomationLoop
    
    loop = GEPAAutomationLoop()
    
    results = {
        "database": await loop._check_database(),
        "api_keys": await loop._check_api_keys(),
        "storage": await loop._check_storage(),
        "gepa_package": await loop._check_gepa_package(),
    }
    
    return {
        "status": "healthy" if all(results.values()) else "degraded",
        "checks": results,
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
