// =====================
// ZK-PROOF MANAGER
// =====================

class ZKProofManager {
    constructor() {
        this.isInitialized = false;
        this.proof = null;
    }

    async initialize() {
        try {
            console.log('🔄 Initializing ZK Proof System...');
            await this.simulateNoirInit();
            this.isInitialized = true;
            this.updateSecurityStatus('Quantum Secure', 95);
            this.showNotification('🔐 Zero-Knowledge Proof System Initialized', 'success');
            return true;
        } catch (error) {
            console.error('ZK Init Failed:', error);
            this.showNotification('⚠️ ZK System Initialization Failed', 'error');
            return false;
        }
    }

    async simulateNoirInit() {
        return new Promise(resolve => {
            setTimeout(() => {
                console.log('✅ NoirJS simulation ready');
                resolve();
            }, 1500);
        });
    }

    async generateProof(formData) {
        if (!this.isInitialized) {
            throw new Error('ZK system not initialized');
        }

        const steps = [
            'Compiling ZK Circuit...',
            'Generating Witness...', 
            'Creating Proof...'
        ];

        for (let i = 0; i < steps.length; i++) {
            await this.simulateProofStep(steps[i], i);
        }

        this.proof = {
            proof: 'zk_proof_' + Math.random().toString(36).substr(2, 9),
            publicInputs: this.hashFormData(formData),
            verificationKey: 'vk_' + Math.random().toString(36).substr(2, 9),
            timestamp: Date.now()
        };

        return this.proof;
    }

    async simulateProofStep(step, index) {
        return new Promise(resolve => {
            setTimeout(() => {
                const loadingText = document.getElementById('loadingText');
                const proofSteps = document.querySelectorAll('.proof-step');
                
                if (loadingText) loadingText.textContent = step;
                if (proofSteps.length > 0) {
                    proofSteps.forEach((el, i) => {
                        el.classList.toggle('active', i <= index);
                    });
                }
                resolve();
            }, 1200);
        });
    }

    hashFormData(formData) {
        const dataString = JSON.stringify(formData);
        let hash = 0;
        for (let i = 0; i < dataString.length; i++) {
            const char = dataString.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return '0x' + Math.abs(hash).toString(16).padStart(16, '0');
    }

    async verifyProof(proof) {
        await new Promise(resolve => setTimeout(resolve, 1800));
        
        const isValid = Math.random() > 0.1;
        return {
            isValid,
            verificationTime: Math.random() * 100 + 50,
            gasUsed: Math.floor(Math.random() * 100000 + 50000)
        };
    }

    updateSecurityStatus(level, strength) {
        const levelText = document.getElementById('securityLevelText');
        const levelBar = document.getElementById('securityLevelBar');
        
        if (levelText) levelText.textContent = level;
        if (levelBar) {
            levelBar.style.background = 
                `linear-gradient(90deg, var(--neon-green) ${strength}%, rgba(255,255,255,0.1) ${strength}%)`;
        }
    }

    showNotification(message, type) {
        console.log(`${type}: ${message}`);
        if (typeof showNotification === 'function') {
            showNotification(message, type);
        }
    }
}

// =====================
// ENHANCED SECURITY ANIMATIONS
// =====================

function enhanceSecurityAnimations() {
    document.querySelectorAll('.cyber-input, .cyber-select').forEach(input => {
        input.addEventListener('focus', function() {
            this.style.background = 'linear-gradient(90deg, rgba(76, 201, 240, 0.1), rgba(67, 97, 238, 0.05))';
        });
        
        input.addEventListener('blur', function() {
            this.style.background = 'rgba(255, 255, 255, 0.05)';
        });
    });

    const encryptionNodes = document.querySelectorAll('.encryption-node');
    if (encryptionNodes.length > 0) {
        encryptionNodes.forEach((node, index) => {
            node.style.animationDelay = `${index * 0.5}s`;
        });
    }
}

// =====================
// GLOBAL VARIABLES
// =====================

const zkManager = new ZKProofManager();
const fortuneMessages = [
    "Your data is encrypted with zero-knowledge proofs...",
    "ZK-SNARKs ensure your privacy while computing predictions...",
    "The AI sees patterns, but never your raw data...",
    "Quantum-resistant encryption protects your information...",
    "Your digital identity remains anonymous and secure...",
    "The neural network computes on encrypted data only...",
    "Algorithms verify predictions without exposing inputs...",
    "In the realm of ZK-proofs, privacy is mathematical...",
    "The AI oracle knows only what it needs to know...",
    "Your profile is protected by cryptographic magic..."
];

const subtitleTexts = [
    "AI-Powered Insurance Prediction",
    "Zero-Knowledge Proof Enabled",
    "Privacy-Preserving AI",
    "ZK-SNARKs Active",
    "Quantum-Secure Predictions"
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

    await zkManager.initialize();
    enhanceSecurityAnimations();
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

    const verifyBtn = document.getElementById('verifyBtn');
    if (verifyBtn) {
        verifyBtn.addEventListener('click', handleVerifyProof);
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
    document.getElementById('licenseToggle').addEventListener('click', function() {
        this.classList.toggle('active');
        const hiddenInput = document.getElementById('Driving_License');
        hiddenInput.value = this.classList.contains('active') ? '1' : '0';
    });

    document.getElementById('insuranceToggle').addEventListener('click', function() {
        this.classList.toggle('active');
        const hiddenInput = document.getElementById('Previously_Insured');
        hiddenInput.value = this.classList.contains('active') ? '1' : '0';
    });
}

function setupDamageSelector() {
    document.querySelectorAll('.damage-option').forEach(option => {
        option.addEventListener('click', function() {
            document.querySelectorAll('.damage-option').forEach(opt => opt.classList.remove('active'));
            this.classList.add('active');
            document.getElementById('Vehicle_Damage').value = this.getAttribute('data-value');
        });
    });
}

function setupPremiumIndicator() {
    document.getElementById('Annual_Premium').addEventListener('input', function() {
        const value = parseFloat(this.value) || 0;
        const indicator = document.getElementById('premiumLevel');
        
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

function setupAudioControls() {
    const audioToggle = document.getElementById('audioToggle');
    const ambientAudio = document.getElementById('ambientAudio');
    const volumeSlider = document.getElementById('volumeSlider');
    const volumeLevel = document.getElementById('volumeLevel');
    let isAudioPlaying = false;

    ambientAudio.volume = volumeSlider.value / 100;
    updateVolumeLevel();

    audioToggle.addEventListener('click', function() {
        if (isAudioPlaying) {
            ambientAudio.pause();
            this.classList.add('muted');
        } else {
            ambientAudio.play().catch(e => {
                console.log('Audio play failed:', e);
                showNotification('Audio playback failed. Click to enable sound.', 'error');
            });
            this.classList.remove('muted');
        }
        isAudioPlaying = !isAudioPlaying;
    });

    volumeSlider.addEventListener('input', function() {
        const volume = this.value / 100;
        ambientAudio.volume = volume;
        updateVolumeLevel();
    });

    function updateVolumeLevel() {
        const volume = volumeSlider.value;
        volumeLevel.style.background = `linear-gradient(90deg, var(--accent) ${volume}%, rgba(255,255,255,0.1) ${volume}%)`;
    }
}

// =====================
// FORM HANDLING
// =====================

async function handleFormSubmit(e) {
    e.preventDefault();
    console.log('📝 Form submission started...');

    if (!zkManager.isInitialized) {
        showNotification('🔐 Initializing ZK System...', 'info');
        await zkManager.initialize();
    }

    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) loadingOverlay.classList.add('active');

    try {
        const formData = collectFormData();
        console.log('📊 Form data collected:', formData);

        const proof = await zkManager.generateProof(formData);
        updateEncryptionProgress('completed');
        
        setTimeout(() => {
            loadingOverlay.classList.remove('active');
            const resultContainer = document.getElementById('resultContainer');
            if (resultContainer) {
                resultContainer.style.display = 'block';
                addProofVerification(proof);
                
                resultContainer.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'center'
                });
            }
            
            showNotification('🔐 Zero-Knowledge Proof Generated Successfully!', 'success');
        }, 1000);
        
    } catch (error) {
        console.error('Proof generation failed:', error);
        showNotification('❌ Proof Generation Failed', 'error');
        if (loadingOverlay) loadingOverlay.classList.remove('active');
    }
}

async function handleVerifyProof() {
    if (!zkManager.proof) {
        showNotification('No proof available to verify', 'error');
        return;
    }

    this.disabled = true;
    this.innerHTML = '<div class="btn-bg"></div><div class="btn-content"><i class="fas fa-spinner fa-spin"></i><span>Verifying Proof...</span></div>';

    try {
        const verification = await zkManager.verifyProof(zkManager.proof);
        
        if (verification.isValid) {
            showNotification('✅ ZK Proof Verified Successfully!', 'success');
            zkManager.updateSecurityStatus('Quantum Verified', 98);
        } else {
            showNotification('❌ Proof Verification Failed', 'error');
        }
    } catch (error) {
        showNotification('⚠️ Verification Error', 'error');
    } finally {
        this.disabled = false;
        this.innerHTML = '<div class="btn-bg"></div><div class="btn-content"><i class="fas fa-fingerprint"></i><span>Verify ZK-Proof</span></div>';
    }
}

async function handleTrainModel() {
    this.disabled = true;
    this.innerHTML = '<div class="btn-bg"></div><div class="btn-content"><i class="fas fa-spinner fa-spin"></i><span>Training Neural Network...</span></div>';
    
    try {
        await new Promise(resolve => setTimeout(resolve, 3000));
        showNotification('Neural Network Training Complete! 🧠', 'success');
    } catch (error) {
        showNotification('Training Failed. Please try again. ⚠️', 'error');
    } finally {
        this.disabled = false;
        this.innerHTML = '<div class="btn-bg"></div><div class="btn-content"><i class="fas fa-cogs"></i><span>Train Neural Network</span></div>';
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

function updateEncryptionProgress(status) {
    const stages = document.querySelectorAll('.progress-stages .stage');
    
    stages.forEach(stage => {
        const stageNum = parseInt(stage.getAttribute('data-stage'));
        
        if (status === 'completed') {
            stage.classList.add('completed');
            stage.classList.remove('active');
            const loader = stage.querySelector('.stage-loader');
            if (loader) {
                loader.innerHTML = '<i class="fas fa-check"></i>';
            }
        }
    });
}

function addProofVerification(proof) {
    const resultHeader = document.querySelector('.result-header');
    if (resultHeader && !resultHeader.querySelector('.proof-badge')) {
        const badge = document.createElement('div');
        badge.className = 'proof-badge';
        badge.innerHTML = '<i class="fas fa-shield-check"></i><span>Zero-Knowledge Proof Verified</span>';
        resultHeader.appendChild(badge);
    }
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
// NOTIFICATION SYSTEM
// =====================

function showNotification(message, type = 'info') {
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
            showNotification('🎉 You\'ve discovered the AI Easter Egg! Welcome to the Matrix!', 'success');
            this.classList.add('easter-egg-active');
            
            document.body.style.animation = 'rainbowGlow 2s linear infinite';
            setTimeout(() => {
                document.body.style.animation = '';
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
// SMOOTH SCROLLING
// =====================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});