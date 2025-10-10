// =====================
// INSURANCE IQ MANAGER
// =====================

class InsuranceIQManager {
    constructor() {
        this.isInitialized = false;
        this.currentPrediction = null;
        this.trainingInterval = null;
    }

    async initialize() {
        try {
            console.log('🚀 Initializing InsuranceIQ AI System...');
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
            setTimeout(() => {
                console.log('✅ AI Neural Network simulation ready');
                resolve();
            }, 1500);
        });
    }

    async processPrediction(formData) {
        if (!this.isInitialized) {
            throw new Error('AI system not initialized');
        }

        const steps = [
            'Analyzing Personal Profile...',
            'Processing Vehicle Data...', 
            'Calculating Risk Factors...',
            'Generating Insurance Score...'
        ];

        for (let i = 0; i < steps.length; i++) {
            await this.simulateProcessingStep(steps[i], i);
        }

        this.currentPrediction = {
            prediction: Math.random() > 0.5 ? 'high' : 'low',
            confidence: Math.random() * 30 + 70,
            factors: this.analyzeFactors(formData),
            timestamp: Date.now()
        };

        return this.currentPrediction;
    }

    async simulateProcessingStep(step, index) {
        return new Promise(resolve => {
            setTimeout(() => {
                const loadingText = document.getElementById('loadingText');
                if (loadingText) loadingText.textContent = step;
                
                const progress = ((index + 1) / 4) * 100;
                const progressBar = document.getElementById('progressBar');
                if (progressBar) progressBar.style.width = `${progress}%`;
                
                resolve();
            }, 1200);
        });
    }

    analyzeFactors(formData) {
        const factors = [];
        if (formData.Vehicle_Damage === 'Yes') factors.push('Vehicle Damage History');
        if (formData.Age < 25) factors.push('Young Driver');
        if (formData.Previously_Insured === '0') factors.push('First-time Insured');
        if (formData.Annual_Premium > 30000) factors.push('High Premium');
        
        return factors.length > 0 ? factors : ['Standard Risk Profile'];
    }

    updateSystemStatus(level, strength) {
        console.log(`System Status: ${level} - ${strength}%`);
    }

    showNotification(message, type) {
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
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    container.removeChild(notification);
                }
            }, 300);
        });
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.classList.remove('show');
                setTimeout(() => {
                    if (notification.parentNode) {
                        container.removeChild(notification);
                    }
                }, 300);
            }
        }, 5000);
    }

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
                this.showNotification('🧠 Neural Network Training Started!', 'info');
                this.monitorTrainingProgress();
            } else {
                this.showNotification('⚠️ ' + data.error, 'error');
            }
        } catch (error) {
            console.error('Training start error:', error);
            this.showNotification('❌ Failed to start training', 'error');
        }
    }

    async monitorTrainingProgress() {
        this.trainingInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/train/status');
                const data = await response.json();
                
                if (data.status === 'training') {
                    this.showTrainingProgress(data);
                } else if (data.status === 'completed') {
                    this.showTrainingComplete();
                    clearInterval(this.trainingInterval);
                } else if (data.status === 'idle') {
                    clearInterval(this.trainingInterval);
                }
            } catch (error) {
                console.error('Progress monitoring error:', error);
                clearInterval(this.trainingInterval);
            }
        }, 1000);
    }

    showTrainingProgress(data) {
        const message = `Training Progress: ${data.progress}% - ${data.message}`;
        this.showNotification(message, 'info');
        
        // Update any progress indicators if they exist
        const progressElements = document.querySelectorAll('.training-progress');
        progressElements.forEach(element => {
            element.style.width = `${data.progress}%`;
        });
    }

    showTrainingComplete() {
        this.showNotification('✅ Neural Network Training Complete!', 'success');
        
        // Celebrate completion with special effects
        this.celebrateTrainingCompletion();
    }

    celebrateTrainingCompletion() {
        // Add some visual celebration
        const celebration = document.createElement('div');
        celebration.innerHTML = '🎉🧠✨';
        celebration.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 4rem;
            z-index: 10000;
            pointer-events: none;
            animation: celebrate 2s ease-out forwards;
        `;
        
        document.body.appendChild(celebration);
        
        setTimeout(() => {
            document.body.removeChild(celebration);
        }, 2000);
    }
}

// =====================
// GLOBAL VARIABLES
// =====================

const insuranceManager = new InsuranceIQManager();
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
    "Your profile resonates with the frequency of fortune..."
];

const subtitleTexts = [
    "AI-Powered Insurance Prediction",
    "Machine Learning at Your Service",
    "Decode Your Insurance Future",
    "Neural Networks Never Lie",
    "Data Science Meets Destiny"
];

let currentSubtitleIndex = 0;
let currentCharIndex = 0;
let isDeleting = false;

// =====================
// INITIALIZATION
// =====================

document.addEventListener('DOMContentLoaded', async function() {
    console.log('🚀 Initializing InsuranceIQ...');
    
    setTimeout(() => {
        document.body.classList.add('loaded');
    }, 500);

    await insuranceManager.initialize();
    setupEventListeners();
    initializeAnimations();
    
    console.log('✅ InsuranceIQ initialized successfully');
});

// =====================
// EVENT LISTENERS
// =====================

function setupEventListeners() {
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handleFormSubmit);
    }

    const trainBtn = document.getElementById('trainBtn');
    if (trainBtn) {
        trainBtn.addEventListener('click', handleTrainModel);
    }

    setupAudioControls();
    setupInteractiveToggles();
    setupDamageSelector();
    setupPremiumIndicator();
}

function setupInteractiveToggles() {
    const licenseToggle = document.getElementById('licenseToggle');
    if (licenseToggle) {
        licenseToggle.addEventListener('click', function() {
            this.classList.toggle('active');
            const hiddenInput = document.getElementById('Driving_License');
            if (hiddenInput) {
                hiddenInput.value = this.classList.contains('active') ? '1' : '0';
            }
        });
    }

    const insuranceToggle = document.getElementById('insuranceToggle');
    if (insuranceToggle) {
        insuranceToggle.addEventListener('click', function() {
            this.classList.toggle('active');
            const hiddenInput = document.getElementById('Previously_Insured');
            if (hiddenInput) {
                hiddenInput.value = this.classList.contains('active') ? '1' : '0';
            }
        });
    }
}

function setupDamageSelector() {
    document.querySelectorAll('.damage-option').forEach(option => {
        option.addEventListener('click', function() {
            document.querySelectorAll('.damage-option').forEach(opt => opt.classList.remove('active'));
            this.classList.add('active');
            const damageInput = document.getElementById('Vehicle_Damage');
            if (damageInput) {
                damageInput.value = this.getAttribute('data-value');
            }
        });
    });
}

function setupPremiumIndicator() {
    const premiumInput = document.getElementById('Annual_Premium');
    if (premiumInput) {
        premiumInput.addEventListener('input', function() {
            const value = parseFloat(this.value) || 0;
            const indicator = document.getElementById('premiumLevel');
            if (!indicator) return;
            
            if (value < 10000) {
                indicator.textContent = 'Basic';
                indicator.style.color = '#4ade80';
            } else if (value < 30000) {
                indicator.textContent = 'Standard';
                indicator.style.color = '#fca311';
            } else {
                indicator.textContent = 'Premium';
                indicator.style.color = '#f72585';
            }
        });
    }
}

function setupAudioControls() {
    const audioToggle = document.getElementById('audioToggle');
    const ambientAudio = document.getElementById('ambientAudio');
    let isAudioPlaying = false;

    if (audioToggle && ambientAudio) {
        audioToggle.addEventListener('click', function() {
            if (isAudioPlaying) {
                ambientAudio.pause();
                this.innerHTML = '<i class="fas fa-volume-mute"></i>';
                this.classList.add('muted');
            } else {
                ambientAudio.play().catch(e => {
                    console.log('Audio play failed:', e);
                });
                this.innerHTML = '<i class="fas fa-volume-up"></i>';
                this.classList.remove('muted');
            }
            isAudioPlaying = !isAudioPlaying;
        });
    }
}

// =====================
// FORM HANDLING
// =====================

async function handleFormSubmit(e) {
    e.preventDefault();
    console.log('📝 Form submission started...');

    if (!insuranceManager.isInitialized) {
        insuranceManager.showNotification('🤖 Initializing AI System...', 'info');
        await insuranceManager.initialize();
    }

    // Show loading overlay
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.classList.add('active');
        
        // Reset progress bar
        const progressBar = document.getElementById('progressBar');
        if (progressBar) progressBar.style.width = '0%';
    }

    try {
        const formData = collectFormData();
        console.log('📊 Form data collected:', formData);

        // Show processing steps
        await insuranceManager.processPrediction(formData);
        
        // Now submit the form via AJAX for better UX
        const response = await fetch('/', {
            method: 'POST',
            body: new FormData(e.target)
        });
        
        if (response.ok) {
            const html = await response.text();
            
            // Hide loading overlay
            if (loadingOverlay) loadingOverlay.classList.remove('active');
            
            // Update the page with new content
            document.open();
            document.write(html);
            document.close();
            
            // Reinitialize everything
            setTimeout(() => {
                setupEventListeners();
                initializeAnimations();
            }, 100);
            
        } else {
            throw new Error('Server response not OK');
        }
        
    } catch (error) {
        console.error('Prediction processing failed:', error);
        insuranceManager.showNotification('❌ Prediction Processing Failed', 'error');
        if (loadingOverlay) loadingOverlay.classList.remove('active');
    }
}

async function handleTrainModel() {
    const trainBtn = document.getElementById('trainBtn');
    if (!trainBtn) return;

    // Disable button and show loading state
    trainBtn.disabled = true;
    const originalContent = trainBtn.innerHTML;
    
    trainBtn.innerHTML = `
        <div class="btn-bg"></div>
        <div class="btn-content">
            <i class="fas fa-spinner fa-spin"></i>
            <span>Training Neural Network...</span>
        </div>
    `;
    
    try {
        await insuranceManager.startTraining();
    } catch (error) {
        console.error('Training error:', error);
        insuranceManager.showNotification('❌ Training Failed. Please try again.', 'error');
    } finally {
        // Restore button
        trainBtn.disabled = false;
        trainBtn.innerHTML = originalContent;
    }
}

// =====================
// HELPER FUNCTIONS
// =====================

function collectFormData() {
    const formData = {};
    const formElements = document.getElementById('predictionForm').elements;
    
    for (let element of formElements) {
        if (element.name && element.value !== '') {
            formData[element.name] = element.value;
        }
    }
    
    return formData;
}

// =====================
// ANIMATIONS & EFFECTS
// =====================

function initializeAnimations() {
    createMatrixRain();
    createParticles();
    initNeuralNetwork();
    startTypingEffect();
    startFortuneRotation();
    initCyberCursor();
    startStatsCounter();
}

function createMatrixRain() {
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

function createParticles() {
    const container = document.getElementById('particlesContainer');
    if (!container) return;
    
    for (let i = 0; i < 100; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 10 + 's';
        particle.style.animationDuration = (Math.random() * 20 + 10) + 's';
        container.appendChild(particle);
    }
}

function initNeuralNetwork() {
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
    
    for (let i = 0; i < 15; i++) {
        nodes.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            radius: Math.random() * 3 + 2
        });
    }
    
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            if (Math.random() > 0.7) {
                connections.push({
                    from: i,
                    to: j,
                    strength: Math.random()
                });
            }
        }
    }
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        nodes.forEach(node => {
            node.x += node.vx;
            node.y += node.vy;
            
            if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
            if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
            
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(76, 201, 240, ${0.7})`;
            ctx.fill();
            
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius * 3, 0, Math.PI * 2);
            const gradient = ctx.createRadialGradient(
                node.x, node.y, node.radius,
                node.x, node.y, node.radius * 3
            );
            gradient.addColorStop(0, 'rgba(76, 201, 240, 0.3)');
            gradient.addColorStop(1, 'rgba(76, 201, 240, 0)');
            ctx.fillStyle = gradient;
            ctx.fill();
        });
        
        connections.forEach(conn => {
            const from = nodes[conn.from];
            const to = nodes[conn.to];
            const distance = Math.sqrt((from.x - to.x) ** 2 + (from.y - to.y) ** 2);
            
            if (distance < 200) {
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = `rgba(76, 201, 240, ${0.2 * conn.strength})`;
                ctx.lineWidth = conn.strength;
                ctx.stroke();
            }
        });
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

function startTypingEffect() {
    function typeWriter() {
        const subtitleElement = document.getElementById('subtitleText');
        if (!subtitleElement) return;
        
        const currentText = subtitleTexts[currentSubtitleIndex];

        if (!isDeleting) {
            subtitleElement.textContent = currentText.substring(0, currentCharIndex + 1);
            currentCharIndex++;

            if (currentCharIndex === currentText.length) {
                isDeleting = true;
                setTimeout(typeWriter, 2000);
            } else {
                setTimeout(typeWriter, 100);
            }
        } else {
            subtitleElement.textContent = currentText.substring(0, currentCharIndex - 1);
            currentCharIndex--;

            if (currentCharIndex === 0) {
                isDeleting = false;
                currentSubtitleIndex = (currentSubtitleIndex + 1) % subtitleTexts.length;
                setTimeout(typeWriter, 500);
            } else {
                setTimeout(typeWriter, 50);
            }
        }
    }

    typeWriter();
}

function startFortuneRotation() {
    let currentFortuneIndex = 0;
    function rotateFortune() {
        const fortuneText = document.getElementById('fortuneText');
        if (!fortuneText) return;
        
        fortuneText.style.opacity = '0';
        
        setTimeout(() => {
            currentFortuneIndex = (currentFortuneIndex + 1) % fortuneMessages.length;
            fortuneText.textContent = fortuneMessages[currentFortuneIndex];
            fortuneText.style.opacity = '1';
        }, 500);
    }

    setInterval(rotateFortune, 4000);
}

function initCyberCursor() {
    const cyberCursor = document.getElementById('cyberCursor');
    if (!cyberCursor) return;
    
    let mouseX = 0, mouseY = 0;
    let cursorX = 0, cursorY = 0;

    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    function animateCursor() {
        const dx = mouseX - cursorX;
        const dy = mouseY - cursorY;
        
        cursorX += dx * 0.1;
        cursorY += dy * 0.1;
        
        cyberCursor.style.transform = `translate(${cursorX}px, ${cursorY}px)`;
        requestAnimationFrame(animateCursor);
    }
    animateCursor();
}

function startStatsCounter() {
    function animateCounter(element, target, duration = 2000) {
        let start = 0;
        const increment = target / (duration / 16);
        
        function updateCounter() {
            start += increment;
            if (start < target) {
                element.textContent = Math.floor(start);
                requestAnimationFrame(updateCounter);
            } else {
                element.textContent = target;
            }
        }
        
        updateCounter();
    }

    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const statNumbers = entry.target.querySelectorAll('.stat-number');
                statNumbers.forEach(stat => {
                    const target = parseFloat(stat.getAttribute('data-target'));
                    animateCounter(stat, target);
                });
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    const statsContainer = document.querySelector('.stats-container');
    if (statsContainer) {
        observer.observe(statsContainer);
    }
}

// =====================
// EASTER EGGS
// =====================

document.addEventListener('DOMContentLoaded', function() {
    const logoHologram = document.querySelector('.logo-hologram');
    if (!logoHologram) return;
    
    let clickCount = 0;
    logoHologram.addEventListener('click', function() {
        clickCount++;
        if (clickCount >= 5) {
            insuranceManager.showNotification('🎉 You\'ve discovered the AI Easter Egg! Welcome to the Matrix!', 'success');
            this.classList.add('easter-egg-active');
            
            setTimeout(() => {
                this.classList.remove('easter-egg-active');
            }, 5000);
            
            clickCount = 0;
        }
    });
});

// =====================
// KEYBOARD SHORTCUTS
// =====================

document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'm') {
        e.preventDefault();
        const audioToggle = document.getElementById('audioToggle');
        if (audioToggle) audioToggle.click();
    }
    
    if (e.key === 'Escape') {
        document.querySelectorAll('.notification.show').forEach(notification => {
            const closeBtn = notification.querySelector('.notification-close');
            if (closeBtn) closeBtn.click();
        });
    }
});

// =====================
// ADDITIONAL STYLES FOR CELEBRATION
// =====================

const celebrationStyles = `
@keyframes celebrate {
    0% {
        transform: translate(-50%, -50%) scale(0) rotate(0deg);
        opacity: 0;
    }
    50% {
        transform: translate(-50%, -50%) scale(1.5) rotate(180deg);
        opacity: 1;
    }
    100% {
        transform: translate(-50%, -50%) scale(1) rotate(360deg);
        opacity: 0;
    }
}
`;

// Add celebration styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = celebrationStyles;
document.head.appendChild(styleSheet);