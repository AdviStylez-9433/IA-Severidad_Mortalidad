document.addEventListener('DOMContentLoaded', function() {
    // Solo para demostración - mantener los datos visibles
    console.log("Pacientes cargados:", document.querySelectorAll('#evaluationsTable tr').length);
    
    // Deshabilitar cualquier código que pueda estar borrando la tabla
    const tableBody = document.getElementById('evaluationsTable');
    if (tableBody) {
        tableBody.style.display = 'table-row-group'; // Forzar visibilidad
    }
});