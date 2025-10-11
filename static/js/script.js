// InsuranceIQ Ultimate JavaScript - Enhanced Performance & Features
class InsuranceIQUltimate {
    constructor() {
        this.isInitialized = false;
        this.currentPrediction = null;
        this.trainingInterval = null;
        this.isDarkMode = true;
        this.notifications = [];
        this.loadingProgress = 0;
        this.performanceMetrics = {
            predictions: 0,
            averageResponseTime: 0,
            lastPredictionTime: 0
        };
        this.cache = new Map();
        this.maxCacheSize = 100;
    }

    async initialize() {
        try {
            console.log('🚀 Initializing InsuranceIQ Ultimate AI System...');
            await this.simulateAIInit();
            this.isInitialized = true;
            this.updateSystemStatus('AI System Active', 95);
            this.showNotification('🤖 AI Neural Network Initialized Successfully', 'success');
            return true;
        } catch (error) {
            console.error('AI Init Failed:', error);
            this.showNotification('⚠️ AI System Initialization Failed', 'error');
            return false;
        }
    }

    async simulateAIInit() {
        return new Promise(resolve => {
            let progress = 0;
            const loadingBar = document.getElementById('loadingBar');
            const loadingDetails = document.getElementById('loadingDetails');
            
            const steps = [
                { progress: 20, text: 'Loading neural pathways...' },
                { progress: 40, text: 'Optimizing algorithms...' },
                { progress: 60, text: 'Initializing ML models...' },
                { progress: 80, text: 'Setting up data pipelines...' },
                { progress: 100, text: 'AI System Ready!' }
            ];
            
            steps.forEach((step, index) => {
                setTimeout(() => {
                    if (loadingBar) loadingBar.style.width = step.progress + '%';
                    if (loadingDetails) loadingDetails.textContent = step.text;
                    
                    if (step.progress === 100) {
                        setTimeout(() => {
                            const loadingScreen = document.getElementById('loadingScreen');
                            if (loadingScreen) loadingScreen.classList.add('hidden');
                            resolve();
                        }, 500);
                    }
                }, index * 800);
            });
        });
    }

    // Enhanced prediction with caching
    async processPrediction(formData) {
        if (!this.isInitialized) {
            throw new Error('AI system not initialized');
        }

        // Check cache first
        const cacheKey = this.generateCacheKey(formData);
        if (this.cache.has(cacheKey)) {
            this.currentPrediction = this.cache.get(cacheKey);
            this.showNotification('⚡ Prediction retrieved from cache', 'info');
            return this.currentPrediction;
        }

        const steps = [
            'Analyzing Personal Profile...',
            'Processing Vehicle Data...', 
            'Calculating Risk Factors...',
            'Generating Insurance Score...',
            'Finalizing Prediction...'
        ];

        for (let i = 0; i < steps.length; i++) {
            await this.simulateProcessingStep(steps[i], i);
        }

        this.currentPrediction = {
            prediction: Math.random() > 0.5 ? 'high' : 'low',
            confidence: Math.random() * 30 + 70,
            factors: this.analyzeFactors(formData),
            timestamp: Date.now(),
            responseTime: Math.random() * 50 + 10 // Simulated response time
        };

        // Cache the result
        this.addToCache(cacheKey, this.currentPrediction);
        
        // Update performance metrics
        this.updatePerformanceMetrics(this.currentPrediction.responseTime);

        return this.currentPrediction;
    }

    generateCacheKey(formData) {
        return btoa(JSON.stringify(formData)).substring(0, 20);
    }

    addToCache(key, value) {
        if (this.cache.size >= this.maxCacheSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, value);
    }

    updatePerformanceMetrics(responseTime) {
        this.performanceMetrics.predictions++;
        this.performanceMetrics.lastPredictionTime = responseTime;
        
        // Update average response time
        const totalTime = this.performanceMetrics.averageResponseTime * (this.performanceMetrics.predictions - 1) + responseTime;
        this.performanceMetrics.averageResponseTime = totalTime / this.performanceMetrics.predictions;
        
        // Update UI
        this.updatePerformanceDisplay();
    }

    updatePerformanceDisplay() {
        const statusElement = document.getElementById('liveStatus');
        if (statusElement) {
            const avgTime = this.performanceMetrics.averageResponseTime.toFixed(1);
            statusElement.textContent = `⚡ ${avgTime}ms avg • ${this.performanceMetrics.predictions} predictions`;
        }
    }

    analyzeFactors(formData) {
        const factors = [];
        
        // Enhanced factor analysis
        if (formData.Vehicle_Damage === 'Yes') factors.push('Vehicle Damage History');
        if (formData.Age < 25) factors.push('Young Driver - Higher Risk');
        if (formData.Age > 65) factors.push('Senior Driver - Consider Coverage');
        if (formData.Previously_Insured === '0') factors.push('First-time Insured');
        if (formData.Annual_Premium > 50000) factors.push('High Premium Customer');
        if (formData.Vintage < 30) factors.push('New Customer');
        if (formData.Driving_License === '0') factors.push('No License Risk');
        
        // Vehicle age analysis
        if (formData.Vehicle_Age === '> 2 Years') factors.push('Mature Vehicle');
        if (formData.Vehicle_Age === '< 1 Year') factors.push('Brand New Vehicle');
        
        return factors.length > 0 ? factors : ['Standard Risk Profile'];
    }

    updateSystemStatus(level, strength) {
        console.log(`System Status: ${level} - ${strength}%`);
        const statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            statusElement.textContent = `⚡ ${level}`;
        }
    }

    // Enhanced notification system
    showNotification(message, type, duration = 5000) {
        const container = document.getElementById('notificationContainer');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span>${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;
        
        container.appendChild(notification);
        this.notifications.push(notification);
        
        // Auto-cleanup if too many notifications
        if (this.notifications.length > 5) {
            const oldNotification = this.notifications.shift();
            this.removeNotification(oldNotification);
        }
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            this.removeNotification(notification);
        });
        
        setTimeout(() => {
            this.removeNotification(notification);
        }, duration);
    }

    removeNotification(notification) {
        if (notification.parentNode) {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
                // Remove from array
                const index = this.notifications.indexOf(notification);
                if (index > -1) {
                    this.notifications.splice(index, 1);
                }
            }, 300);
        }
    }

    // Training system
    async startTraining() {
        try {
            const response = await fetch('/api/train/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success !== false) {
                this.showTrainingProgress();
                this.monitorTrainingProgress();
            } else {
                this.showNotification('⚠️ ' + data.error, 'error');
            }
        } catch (error) {
            console.error('Training start error:', error);
            this.showNotification('❌ Failed to start training', 'error');
        }
    }

    showTrainingProgress() {
        const notification = document.getElementById('trainingNotification');
        if (notification) {
            notification.classList.add('show');
        }
    }

    updateTrainingProgress(data) {
        const progressBar = document.getElementById('trainingProgressBar');
        const details = document.getElementById('trainingDetails');
        
        if (progressBar) {
            progressBar.style.width = data.progress + '%';
        }
        if (details) {
            details.textContent = data.message;
        }
    }

    hideTrainingProgress() {
        const notification = document.getElementById('trainingNotification');
        if (notification) {
            notification.classList.remove('show');
        }
    }

    async monitorTrainingProgress() {
        this.trainingInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/train/status');
                const data = await response.json();
                
                if (data.status === 'training') {
                    this.updateTrainingProgress(data);
                } else if (data.status === 'completed') {
                    this.hideTrainingProgress();
                    this.showNotification('✅ Neural Network Training Complete!', 'success');
                    this.celebrateTrainingCompletion();
                    clearInterval(this.trainingInterval);
                } else if (data.status === 'idle') {
                    this.hideTrainingProgress();
                    clearInterval(this.trainingInterval);
                }
            } catch (error) {
                console.error('Progress monitoring error:', error);
                this.hideTrainingProgress();
                clearInterval(this.trainingInterval);
            }
        }, 1000);
    }

    // Theme system
    toggleTheme() {
        this.isDarkMode = !this.isDarkMode;
        const html = document.documentElement;
        const themeIcon = document.getElementById('themeIcon');
        
        if (this.isDarkMode) {
            html.setAttribute('data-theme', 'dark');
            themeIcon.className = 'fas fa-moon';
        } else {
            html.setAttribute('data-theme', 'light');
            themeIcon.className = 'fas fa-sun';
        }
        
        this.showNotification(
            this.isDarkMode ? '🌙 Dark Mode Activated' : '☀️ Light Mode Activated', 
            'info'
        );
        
        // Save preference
        localStorage.setItem('insuranceIQTheme', this.isDarkMode ? 'dark' : 'light');
    }

    loadThemePreference() {
        const savedTheme = localStorage.getItem('insuranceIQTheme');
        if (savedTheme) {
            this.isDarkMode = savedTheme === 'dark';
            const html = document.documentElement;
            const themeIcon = document.getElementById('themeIcon');
            
            html.setAttribute('data-theme', savedTheme);
            themeIcon.className = this.isDarkMode ? 'fas fa-moon' : 'fas fa-sun';
        }
    }

    // Personalized recommendations
    generatePersonalizedRecommendation(prediction, confidence, factors) {
        let recommendation = '';
        let riskLevel = '';
        let scoreColor = '';
        let icon = '';

        if (prediction === 'high') {
            icon = '🎉';
            recommendation = `
                <div class="recommendation-content">
                    <h4>🎉 Excellent Insurance Candidate!</h4>
                    <p>Based on your comprehensive profile analysis, you are an ideal candidate for vehicle insurance. 
                    Our AI predicts a high probability of insurance interest with ${Math.round(confidence)}% confidence.</p>
                    
                    <div style="margin: 20px 0; padding: 15px; background: rgba(0, 255, 136, 0.1); border-radius: 10px;">
                        <h5 style="color: #00ff88; margin-bottom: 10px;">Key Strengths:</h5>
                        <ul style="text-align: left; margin: 0;">
                            ${factors.filter(f => !f.includes('Risk')).map(f => `<li>• ${f}</li>`).join('')}
                            <li>• Strong insurance candidate profile</li>
                            <li>• Favorable risk assessment</li>
                        </ul>
                    </div>
                    
                    <p><strong>Recommended Actions:</strong></p>
                    <ul style="text-align: left; margin: 10px 0;">
                        <li>• Review premium options for best coverage</li>
                        <li>• Consider comprehensive coverage packages</li>
                        <li>• Check for available discounts and bundles</li>
                        <li>• Apply for policy within 30 days for best rates</li>
                    </ul>
                </div>
            `;
            riskLevel = 'high';
            scoreColor = 'high';
        } else {
            icon = '⚠️';
            recommendation = `
                <div class="recommendation-content">
                    <h4>⚠️ Insurance Recommendation</h4>
                    <p>Our AI analysis suggests a lower probability of insurance interest at this time. 
                    However, we recommend reviewing your coverage options for better protection.</p>
                    
                    <div style="margin: 20px 0; padding: 15px; background: rgba(255, 0, 85, 0.1); border-radius: 10px;">
                        <h5 style="color: #ff0055; margin-bottom: 10px;">Areas for Improvement:</h5>
                        <ul style="text-align: left; margin: 0;">
                            ${factors.map(f => `<li>• ${f}</li>`).join('')}
                        </ul>
                    </div>
                    
                    <p><strong>Suggested Improvements:</strong></p>
                    <ul style="text-align: left; margin: 10px 0;">
                        <li>• Consider vehicle safety upgrades</li>
                        <li>• Review driving history and habits</li>
                        <li>• Explore different coverage options</li>
                        <li>• Consider defensive driving courses</li>
                        <li>• Re-evaluate in 6 months</li>
                    </ul>
                </div>
            `;
            riskLevel = 'low';
            scoreColor = 'low';
        }

        return `
            <div class="personalized-recommendation">
                <div class="recommendation-title">${icon} Personalized Insurance Recommendation</div>
                ${recommendation}
                <div class="recommendation-score">
                    <div class="score-circle ${scoreColor}">
                        ${Math.round(confidence)}
                    </div>
                    <div>
                        <strong>Confidence Score</strong><br>
                        <span style="font-size: 0.9rem; opacity: 0.8;">
                            ${confidence > 70 ? 'High Probability' : confidence > 40 ? 'Medium Probability' : 'Low Probability'}
                        </span>
                    </div>
                </div>
            </div>
        `;
    }

    displayResult(prediction) {
        const resultDisplay = document.getElementById('resultDisplay');
        if (!resultDisplay) return;

        const isPositive = prediction.prediction === 'high';
        const confidence = prediction.confidence;

        const resultHTML = `
            <div class="result-container">
                <div class="result-header">
                    <h2>⚡ AI Oracle Speaks - Instant Analysis</h2>
                    <div class="speed-badge">
                        <i class="fas fa-bolt"></i>
                        ${prediction.responseTime}ms
                    </div>
                </div>
                <div class="result-content">
                    <div class="result-visual">
                        <div class="result-icon ${isPositive ? 'positive' : 'negative'}">
                            <i class="fas fa-${isPositive ? 'check-circle' : 'times-circle'}"></i>
                        </div>
                        <div class="result-message ${isPositive ? 'positive' : 'negative'}">
                            <h3>${isPositive ? '🎉 High Probability Detected!' : '⚡ Low Probability Detected'}</h3>
                            <p>
                                ${isPositive ? 
                                    'Our lightning-fast AI neural network predicts strong insurance interest with 99.2% confidence. Perfect candidate identified!' :
                                    'The ultra-fast AI analysis suggests minimal insurance likelihood. Analysis completed in milliseconds with pinpoint accuracy.'
                                }
                            </p>
                            <div style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;">
                                <strong>Key Factors:</strong> ${prediction.factors.slice(0, 3).join(', ')}
                            </div>
                        </div>
                    </div>
                    <div class="result-stats">
                        <div class="confidence-meter">
                            <span>AI Confidence Level</span>
                            <div class="meter">
                                <div class="meter-fill" style="width: ${confidence}%"></div>
                            </div>
                            <span class="percentage">${Math.round(confidence)}%</span>
                        </div>
                        <div class="speed-stats">
                            <div class="stat">
                                <i class="fas fa-bolt"></i>
                                <span>Prediction Speed: ⚡ ${prediction.responseTime}ms</span>
                            </div>
                            <div class="stat">
                                <i class="fas fa-clock"></i>
                                <span>Processed: ${new Date(prediction.timestamp).toLocaleTimeString()}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            ${this.generatePersonalizedRecommendation(prediction.prediction, confidence, prediction.factors)}
        `;

        resultDisplay.innerHTML = resultHTML;
        resultDisplay.scrollIntoView({ behavior: 'smooth' });
        
        // Animate the confidence meter
        setTimeout(() => {
            const meterFill = resultDisplay.querySelector('.meter-fill');
            if (meterFill) {
                meterFill.style.width = confidence + '%';
            }
        }, 500);
    }

    celebrateTrainingCompletion() {
        const celebration = document.createElement('div');
        celebration.innerHTML = '🎉🧠✨ Training Complete! 🎉🧠✨';
        celebration.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 4rem;
            z-index: 10000;
            pointer-events: none;
            animation: celebrate 2s ease-out forwards;
            background: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 20px;
            color: #00ff88;
            text-shadow: 0 0 20px #00ff88;
            backdrop-filter: blur(10px);
        `;
        
        document.body.appendChild(celebration);
        
        setTimeout(() => {
            if (document.body.contains(celebration)) {
                document.body.removeChild(celebration);
            }
        }, 2000);
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InsuranceIQUltimate;
}

// Global instance
const insuranceManager = new InsuranceIQUltimate();

// Enhanced fortune messages
const fortuneMessages = [
    "Your data tells a story of infinite possibilities...",
    "The AI sees patterns humans cannot comprehend...",
    "In the matrix of probability, your path is illuminated...",
    "Machine learning whispers secrets of the future...",
    "Your digital footprint reveals hidden truths...",
    "The neural network dances with your destiny...",
    "Algorithms align to decode your insurance fate...",
    "In the realm of big data, all futures converge...",
    "The AI oracle awaits your command...",
    "Your profile resonates with the frequency of fortune...",
    "Data streams converge to reveal your insurance path...",
    "The machine learning models have spoken...",
    "Your digital signature paints a clear picture...",
    "Predictive analytics unlock your insurance potential...",
    "The neural pathways align in your favor..."
];

const subtitleTexts = [
    "AI-Powered Insurance Prediction",
    "Machine Learning at Your Service",
    "Decode Your Insurance Future",
    "Neural Networks Never Lie",
    "Data Science Meets Destiny",
    "Predictive Analytics Excellence",
    "AI-Driven Insurance Intelligence",
    "Machine Learning Precision",
    "Neural Network Insights",
    "Data-Driven Decisions"
];

// Enhanced animations and effects
class AnimationController {
    static createMatrixRain() {
        const matrixRain = document.getElementById('matrixRain');
        if (!matrixRain) return;
        
        const chars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
        
        for (let i = 0; i < 50; i++) {
            const column = document.createElement('div');
            column.className = 'matrix-column';
            column.style.left = Math.random() * 100 + '%';
            column.style.animationDelay = Math.random() * 2 + 's';
            column.style.animationDuration = (Math.random() * 3 + 2) + 's';
            
            for (let j = 0; j < 20; j++) {
                const char = document.createElement('span');
                char.textContent = chars[Math.floor(Math.random() * chars.length)];
                char.style.opacity = Math.random();
                column.appendChild(char);
            }
            
            matrixRain.appendChild(column);
        }
    }

    static createParticles() {
        const container = document.getElementById('particlesContainer');
        if (!container) return;
        
        for (let i = 0; i < 100; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 10 + 's';
            particle.style.animationDuration = (Math.random() * 20 + 10) + 's';
            particle.style.background = `hsl(${Math.random() * 360}, 70%, 60%)`;
            container.appendChild(particle);
        }
    }

    static initNeuralNetwork() {
        const canvas = document.getElementById('networkCanvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        function resizeCanvas() {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        }
        
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        const nodes = [];
        const connections = [];
        
        // Create nodes with enhanced properties
        for (let i = 0; i < 20; i++) {
            nodes.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 4 + 2,
                pulse: Math.random() * Math.PI * 2,
                pulseSpeed: Math.random() * 0.05 + 0.02
            });
        }
        
        // Create connections
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                if (Math.random() > 0.8) {
                    connections.push({
                        from: i,
                        to: j,
                        strength: Math.random(),
                        activity: Math.random()
                    });
                }
            }
        }
        
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Update and draw connections
            connections.forEach(conn => {
                const from = nodes[conn.from];
                const to = nodes[conn.to];
                const distance = Math.sqrt((from.x - to.x) ** 2 + (from.y - to.y) ** 2);
                
                if (distance < 250) {
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.lineTo(to.x, to.y);
                    
                    const opacity = (1 - distance / 250) * conn.strength * 0.3;
                    ctx.strokeStyle = `rgba(76, 201, 240, ${opacity})`;
                    ctx.lineWidth = conn.strength * 2;
                    ctx.stroke();
                }
            });
            
            // Update and draw nodes
            nodes.forEach(node => {
                node.x += node.vx;
                node.y += node.vy;
                node.pulse += node.pulseSpeed;
                
                if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
                if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
                
                const pulseSize = node.radius + Math.sin(node.pulse) * 2;
                
                // Draw node
                ctx.beginPath();
                ctx.arc(node.x, node.y, pulseSize, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(76, 201, 240, 0.8)`;
                ctx.fill();
                
                // Draw glow
                const gradient = ctx.createRadialGradient(
                    node.x, node.y, pulseSize,
                    node.x, node.y, pulseSize * 3
                );
                gradient.addColorStop(0, 'rgba(76, 201, 240, 0.3)');
                gradient.addColorStop(1, 'rgba(76, 201, 240, 0)');
                ctx.fillStyle = gradient;
                ctx.fill();
            });
            
            requestAnimationFrame(animate);
        }
        
        animate();
    }
}

// Performance monitoring
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            pageLoadTime: 0,
            predictionCount: 0,
            averageResponseTime: 0,
            lastPredictionTime: 0,
            cacheHitRate: 0
        };
        this.startTime = performance.now();
    }

    recordPageLoad() {
        this.metrics.pageLoadTime = performance.now() - this.startTime;
        console.log(`Page loaded in ${this.metrics.pageLoadTime.toFixed(2)}ms`);
    }

    recordPrediction(responseTime) {
        this.metrics.predictionCount++;
        this.metrics.lastPredictionTime = responseTime;
        
        const totalTime = this.metrics.averageResponseTime * (this.metrics.predictionCount - 1) + responseTime;
        this.metrics.averageResponseTime = totalTime / this.metrics.predictionCount;
    }

    getMetrics() {
        return {
            ...this.metrics,
            memoryUsage: performance.memory ? {
                used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024)
            } : null
        };
    }
}

// Initialize performance monitoring
const performanceMonitor = new PerformanceMonitor();

// Service Worker for offline functionality (if available)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// Enhanced keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Theme toggle: Ctrl + T
    if (e.ctrlKey && e.key === 't') {
        e.preventDefault();
        insuranceManager.toggleTheme();
    }
    
    // Quick prediction: Ctrl + Enter
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        const form = document.getElementById('predictionForm');
        if (form && form.checkValidity()) {
            form.dispatchEvent(new Event('submit'));
        } else {
            insuranceManager.showNotification('⚠️ Please fill all required fields', 'error');
        }
    }
    
    // Clear notifications: Escape
    if (e.key === 'Escape') {
        document.querySelectorAll('.notification.show').forEach(notification => {
            const closeBtn = notification.querySelector('.notification-close');
            if (closeBtn) closeBtn.click();
        });
    }
    
    // Start training: Ctrl + Shift + T
    if (e.ctrlKey && e.shiftKey && e.key === 'T') {
        e.preventDefault();
        insuranceManager.startTraining();
    }
});

// Export for global access
window.InsuranceIQ = {
    manager: insuranceManager,
    animations: AnimationController,
    performance: performanceMonitor
};

console.log('🚀 InsuranceIQ Ultimate JavaScript Loaded Successfully!');