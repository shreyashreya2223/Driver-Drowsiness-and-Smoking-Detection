from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
from driver_monitor import generate_frames, get_stats
import os
import re
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'driver_monitoring_secret_2024'

# Credentials
VALID_CREDENTIALS = {
    'admin': 'driver123'
}

# =====================================
# LOG PARSING FUNCTIONS
# =====================================

def parse_logs():
    """Parse log file and return list of log entries"""
    logs = []
    log_file = "../log.txt"
    
    if not os.path.exists(log_file):
        return logs
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: [YYYY-MM-DD HH:MM:SS] Message
                match = re.match(r'\[([^\]]+)\]\s+(.+)', line)
                if match:
                    timestamp = match.group(1)
                    message = match.group(2)
                    
                    # Determine alert type
                    if 'Drowsiness' in message:
                        alert_type = 'Drowsiness'
                    elif 'Smoking' in message:
                        alert_type = 'Smoking'
                    else:
                        alert_type = 'Other'
                    
                    logs.append({
                        'timestamp': timestamp,
                        'message': message,
                        'type': alert_type
                    })
    except Exception as e:
        print(f"Error parsing logs: {e}")
    
    return logs

def get_log_stats():
    """Calculate statistics from logs"""
    logs = parse_logs()
    
    drowsy_count = sum(1 for log in logs if log['type'] == 'Drowsiness')
    smoking_count = sum(1 for log in logs if log['type'] == 'Smoking')
    total_count = len(logs)
    
    # Calculate alert rate (alerts per total)
    rate = 0
    if total_count > 0:
        rate = int((drowsy_count + smoking_count) / total_count * 100)
    
    return {
        'total': total_count,
        'drowsy': drowsy_count,
        'smoking': smoking_count,
        'rate': rate
    }

def get_log_trend():
    """Build a simple line chart trend from recent logs."""
    logs = parse_logs()
    trend_data = {
        'labels': [],
        'drowsy': [],
        'smoking': []
    }
    
    if not logs:
        return trend_data
    
    recent_logs = logs[-12:]
    counts = []
    
    for log in recent_logs:
        timestamp = log['timestamp']
        time_label = timestamp[-8:]
        
        if counts and counts[-1]['label'] == time_label:
            counts[-1]['drowsy'] = 1 if counts[-1]['drowsy'] or log['type'] == 'Drowsiness' else 0
            counts[-1]['smoking'] = 1 if counts[-1]['smoking'] or log['type'] == 'Smoking' else 0
        else:
            counts.append({
                'label': time_label,
                'drowsy': 1 if log['type'] == 'Drowsiness' else 0,
                'smoking': 1 if log['type'] == 'Smoking' else 0
            })
    
    for item in counts:
        trend_data['labels'].append(item['label'])
        trend_data['drowsy'].append(item['drowsy'])
        trend_data['smoking'].append(item['smoking'])
    
    return trend_data

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/logs')
def logs():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('logs.html')

@app.route('/video_feed')
def video_feed():
    if 'logged_in' not in session:
        return 'Unauthorized', 401

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/stats')
def api_stats():
    if 'logged_in' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(get_stats())

@app.route('/api/logs/recent')
def api_recent_logs():
    if 'logged_in' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    logs = parse_logs()
    recent_logs = logs[-5:] if len(logs) > 5 else logs
    recent_logs.reverse()  # Show newest first
    
    stats = get_log_stats()
    trend = get_log_trend()
    
    return jsonify({
        'logs': recent_logs,
        'stats': stats,
        'trend': trend
    })

@app.route('/api/logs/all')
def api_all_logs():
    if 'logged_in' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    logs = parse_logs()
    logs.reverse()  # Show newest first
    
    return jsonify({'logs': logs})

if __name__ == '__main__':
    app.run(debug=True)