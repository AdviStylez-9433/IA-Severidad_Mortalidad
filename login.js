document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const togglePassword = document.getElementById('togglePassword');
    const passwordInput = document.getElementById('password');
    const loginButton = document.getElementById('loginButton');
    const loginText = document.getElementById('loginText');
    const loginSpinner = document.getElementById('loginSpinner');
    const loginError = document.getElementById('loginError');
    const errorMessage = document.getElementById('errorMessage');

    // Toggle para mostrar/ocultar contraseña
    togglePassword.addEventListener('click', function() {
        const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
        passwordInput.setAttribute('type', type);
        this.querySelector('i').classList.toggle('bi-eye-fill');
        this.querySelector('i').classList.toggle('bi-eye-slash-fill');
    });

    // Manejo del formulario de login
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Simular proceso de autenticación
        loginText.textContent = 'Verificando...';
        loginSpinner.classList.remove('d-none');
        loginButton.disabled = true;
        loginError.classList.add('d-none');

        // Obtener valores del formulario
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const rememberMe = document.getElementById('rememberMe').checked;

        // Simular retraso de red
        setTimeout(() => {
            // Validación simple (en producción esto sería una llamada a la API)
            if (username === 'admin' && password === 'admin123') {
                // Guardar en localStorage si marcó "Recordar"
                if (rememberMe) {
                    localStorage.setItem('rememberedUser', username);
                } else {
                    localStorage.removeItem('rememberedUser');
                }
                
                // Redirigir al dashboard
                window.location.href = 'predict.html';
            } else {
                // Mostrar error
                errorMessage.textContent = 'Usuario o contraseña incorrectos';
                loginError.classList.remove('d-none');
                
                // Restaurar botón
                loginText.textContent = 'Ingresar';
                loginSpinner.classList.add('d-none');
                loginButton.disabled = false;
                
                // Agitar el formulario para feedback
                loginForm.classList.add('shake');
                setTimeout(() => {
                    loginForm.classList.remove('shake');
                }, 500);
            }
        }, 1500);
    });

    // Cargar usuario recordado si existe
    const rememberedUser = localStorage.getItem('rememberedUser');
    if (rememberedUser) {
        document.getElementById('username').value = rememberedUser;
        document.getElementById('rememberMe').checked = true;
    }
});