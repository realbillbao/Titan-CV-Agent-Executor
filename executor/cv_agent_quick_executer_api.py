import os
import sys
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cv_agent_quick_executer import mode_switch
from ap_config import *

from flask import Flask, request, jsonify
app = Flask(__name__)

tasks = {}
lock = threading.Lock()

def execute_mode_switch(query, mode, expr_id, retry_limit, task_id):
    try:
        with lock:
            tasks[task_id] = {"status": "running"}
        result = mode_switch(query=query, mode=mode, expr_id=expr_id, retry_limit=retry_limit)
        with lock:
            tasks[task_id] = {"status": "completed", "result": result}
    except Exception as e:
        with lock:
            tasks[task_id] = {"status": "failed", "error": str(e)}


@app.route('/cv_agent_quick_execute', methods=['POST'])
async def cv_agent_quick_execute():
    data = request.json
    query = data.get('query')
    mode = data.get('mode')
    retry_limit = data.get('retry_limit')

    
    expr_id = data.get('expr_id')
    if expr_id is None or expr_id==0 or expr_id=="":
        from datetime import datetime
        now = datetime.now()
        expr_id = now.strftime('%Y%m%d%H%M%S') + f"{int(now.microsecond / 1000):03d}"

    task_id = expr_id #str(uuid.uuid4())

    thread = threading.Thread(target=execute_mode_switch, args=(query, mode, expr_id, retry_limit, task_id))
    thread.start()
    return jsonify({"task_id": task_id, "status": "started"}), 202


@app.route('/get_quick_execute_state/<task_id>', methods=['GET'])
def get_quick_execute_state(task_id):
    with lock:
        task_state = tasks.get(task_id)
    if task_state:
        return jsonify({"task_id": task_id, "state": task_state})
    else:
        return jsonify({"error": "Task not found"}), 404


if __name__ == '__main__':
    app.run(host = EXECUTOR_FLASK_ADDR, port = EXECUTOR_FLASK_PORT, debug = False)
