console.log("AI Driver Monitoring System Running...");

let running = true;
let score = 100;
let drowsyAlerts = 0;
let smokeAlerts = 0;

// Track previous values to detect new alerts
let previousDrowsy = 0;
let previousSmoke = 0;

/* START */
function startVideo(){
    if(!running){
        document.getElementById("videoFeed").src =
        "/video_feed?" + new Date().getTime();
        running = true;
    }
}

/* STOP */
function stopVideo(){
    document.getElementById("videoFeed").src = "";
    running = false;
}

/* DARK MODE */
function toggleMode(){
    document.body.classList.toggle("light");
}

/* FETCH LIVE STATS AND UPDATE SCORE */
function updateStats(){
    if(running){
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById("fpsValue").innerText = data.fps;
                document.getElementById("earValue").innerText = data.ear;
                document.getElementById("drowsyCount").innerText = data.drowsy_count;
                document.getElementById("smokeCount").innerText = data.smoking_count;
                
                // ====== UPDATE SCORE BASED ON ALERTS ======
                let drowsyIncrease = data.drowsy_count - previousDrowsy;
                let smokeIncrease = data.smoking_count - previousSmoke;
                
                // Drowsiness alert: reduce score by 10
                if(drowsyIncrease > 0){
                    score -= (drowsyIncrease * 10);
                }
                
                // Smoking alert: reduce score by 15
                if(smokeIncrease > 0){
                    score -= (smokeIncrease * 15);
                }
                
                // No new alerts: slowly recover score
                if(drowsyIncrease === 0 && smokeIncrease === 0){
                    if(score < 100){
                        score += 0.05;
                    }
                }
                
                // Keep score within bounds
                if(score < 0) score = 0;
                if(score > 100) score = 100;
                
                // Update UI
                document.getElementById("score").innerText = Math.round(score);
                
                // Update tracking variables
                previousDrowsy = data.drowsy_count;
                previousSmoke = data.smoking_count;
                
                // Change score color based on value
                updateScoreColor();
            })
            .catch(error => console.log('Error fetching stats:', error));
    }
}

/* UPDATE SCORE COLOR */
function updateScoreColor(){
    const scoreElement = document.getElementById("score");
    if(score >= 80){
        scoreElement.style.color = "#22c55e"; // Green
    } else if(score >= 60){
        scoreElement.style.color = "#eab308"; // Yellow
    } else if(score >= 40){
        scoreElement.style.color = "#f97316"; // Orange
    } else {
        scoreElement.style.color = "#ef4444"; // Red
    }
}

/* UPDATE STATS EVERY 500MS */
setInterval(() => {
    updateStats();
}, 500);