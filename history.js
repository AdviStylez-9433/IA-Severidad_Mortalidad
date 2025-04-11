// Simulación de datos almacenados (en un caso real, estos vendrían de una API o localStorage)
let evaluations = JSON.parse(localStorage.getItem('medpredictEvaluations')) || [];

// Elementos del DOM
const evaluationsTable = document.getElementById('evaluationsTable');
const pagination = document.getElementById('pagination');
const totalEvaluationsSpan = document.getElementById('totalEvaluations');
const searchNameInput = document.getElementById('searchName');
const searchDateInput = document.getElementById('searchDate');
const searchRiskSelect = document.getElementById('searchRisk');
const searchButton = document.getElementById('searchButton');
const detailModal = new bootstrap.Modal(document.getElementById('detailModal'));
const detailModalContent = document.getElementById('detailModalContent');
const generatePdfFromHistory = document.getElementById('generatePdfFromHistory');

// Variables de paginación
const itemsPerPage = 10;
let currentPage = 1;
let filteredEvaluations = [];

// Inicialización
document.addEventListener('DOMContentLoaded', () => {
    loadEvaluations();
    setupEventListeners();
});

function loadEvaluations() {
    filteredEvaluations = [...evaluations];
    updateTable();
    updatePagination();
    totalEvaluationsSpan.textContent = evaluations.length;
}

function setupEventListeners() {
    searchButton.addEventListener('click', applyFilters);
    generatePdfFromHistory.addEventListener('click', generatePdf);
}

function applyFilters() {
    const nameFilter = searchNameInput.value.toLowerCase();
    const dateFilter = searchDateInput.value;
    const riskFilter = searchRiskSelect.value;

    filteredEvaluations = evaluations.filter(eval => {
        const matchesName = eval.firstName.toLowerCase().includes(nameFilter) || 
                          eval.lastName.toLowerCase().includes(nameFilter);
        const matchesDate = !dateFilter || eval.date === dateFilter;
        const matchesRisk = !riskFilter || eval.riskLevel === riskFilter;
        
        return matchesName && matchesDate && matchesRisk;
    });

    currentPage = 1;
    updateTable();
    updatePagination();
    totalEvaluationsSpan.textContent = filteredEvaluations.length;
}

function updateTable() {
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const paginatedEvaluations = filteredEvaluations.slice(startIndex, endIndex);

    evaluationsTable.innerHTML = '';

    if (paginatedEvaluations.length === 0) {
        evaluationsTable.innerHTML = `
            <tr>
                <td colspan="6" class="text-center py-4">
                    <i class="bi bi-exclamation-circle me-2"></i>
                    No se encontraron evaluaciones
                </td>
            </tr>
        `;
        return;
    }

    paginatedEvaluations.forEach(evaluation => {
        const row = document.createElement('tr');
        row.className = 'evaluation-item';
        row.innerHTML = `
            <td>
                <strong>${evaluation.lastName}, ${evaluation.firstName}</strong>
            </td>
            <td>${evaluation.age} años</td>
            <td>${formatDate(evaluation.date)}</td>
            <td>
                <span class="badge ${getRiskBadgeClass(evaluation.riskLevel)}">
                    ${getRiskText(evaluation.riskLevel)}
                </span>
            </td>
            <td>
                <span class="severity-indicator severity-${evaluation.severityLevel}">
                    ${evaluation.severityLevel}
                </span>
            </td>
            <td>
                <button class="btn btn-sm btn-outline-hospital-red view-detail" data-id="${evaluation.id}">
                    <i class="bi bi-eye me-1"></i>Ver
                </button>
            </td>
        `;
        evaluationsTable.appendChild(row);
    });

    // Agregar event listeners a los botones de ver detalle
    document.querySelectorAll('.view-detail').forEach(button => {
        button.addEventListener('click', () => showEvaluationDetail(button.dataset.id));
    });
}

function updatePagination() {
    const totalPages = Math.ceil(filteredEvaluations.length / itemsPerPage);
    
    pagination.innerHTML = '';

    if (totalPages <= 1) return;

    // Botón Anterior
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    prevLi.innerHTML = `
        <a class="page-link" href="#" aria-label="Previous">
            <span aria-hidden="true">&laquo;</span>
        </a>
    `;
    prevLi.addEventListener('click', (e) => {
        e.preventDefault();
        if (currentPage > 1) {
            currentPage--;
            updateTable();
            updatePagination();
        }
    });
    pagination.appendChild(prevLi);

    // Números de página
    for (let i = 1; i <= totalPages; i++) {
        const pageLi = document.createElement('li');
        pageLi.className = `page-item ${i === currentPage ? 'active' : ''}`;
        pageLi.innerHTML = `<a class="page-link" href="#">${i}</a>`;
        pageLi.addEventListener('click', (e) => {
            e.preventDefault();
            currentPage = i;
            updateTable();
            updatePagination();
        });
        pagination.appendChild(pageLi);
    }

    // Botón Siguiente
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    nextLi.innerHTML = `
        <a class="page-link" href="#" aria-label="Next">
            <span aria-hidden="true">&raquo;</span>
        </a>
    `;
    nextLi.addEventListener('click', (e) => {
        e.preventDefault();
        if (currentPage < totalPages) {
            currentPage++;
            updateTable();
            updatePagination();
        }
    });
    pagination.appendChild(nextLi);
}

function showEvaluationDetail(id) {
    const evaluation = evaluations.find(e => e.id === id);
    if (!evaluation) return;

    detailModalContent.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6 class="text-hospital-red">Datos del Paciente</h6>
                <ul class="list-group list-group-flush mb-3">
                    <li class="list-group-item d-flex justify-content-between">
                        <span>Nombre completo:</span>
                        <strong>${evaluation.lastName}, ${evaluation.firstName}</strong>
                    </li>
                    <li class="list-group-item d-flex justify-content-between">
                        <span>Edad:</span>
                        <strong>${evaluation.age} años</strong>
                    </li>
                    <li class="list-group-item d-flex justify-content-between">
                        <span>Fecha de evaluación:</span>
                        <strong>${formatDate(evaluation.date)}</strong>
                    </li>
                </ul>
            </div>
            <div class="col-md-6">
                <h6 class="text-hospital-red">Resultados</h6>
                <ul class="list-group list-group-flush mb-3">
                    <li class="list-group-item d-flex justify-content-between">
                        <span>Nivel de riesgo:</span>
                        <span class="badge ${getRiskBadgeClass(evaluation.riskLevel)}">
                            ${getRiskText(evaluation.riskLevel)}
                        </span>
                    </li>
                    <li class="list-group-item d-flex justify-content-between">
                        <span>Nivel de severidad:</span>
                        <span class="severity-indicator severity-${evaluation.severityLevel}">
                            ${evaluation.severityLevel}
                        </span>
                    </li>
                    <li class="list-group-item d-flex justify-content-between">
                        <span>Probabilidad de mortalidad:</span>
                        <strong>${evaluation.mortalityRisk}%</strong>
                    </li>
                </ul>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-12">
                <h6 class="text-hospital-red">Datos Clínicos</h6>
                <div class="table-responsive">
                    <table class="table table-bordered">
                        <thead class="bg-light">
                            <tr>
                                <th>Presión Arterial</th>
                                <th>Frecuencia Cardíaca</th>
                                <th>Sat. Oxígeno</th>
                                <th>Condiciones Crónicas</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>${evaluation.bloodPressure} mmHg</td>
                                <td>${evaluation.heartRate} lpm</td>
                                <td>${evaluation.oxygenLevel}%</td>
                                <td>${getChronicConditionsText(evaluation.chronicConditions)}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-12">
                <h6 class="text-hospital-red">Factores de Riesgo</h6>
                <div class="alert ${getRiskAlertClass(evaluation.riskLevel)}">
                    ${evaluation.riskFactors || 'No se identificaron factores de riesgo adicionales.'}
                </div>
            </div>
        </div>
    `;

    // Almacenar el ID de la evaluación actual para el PDF
    generatePdfFromHistory.dataset.id = id;
    detailModal.show();
}

function generatePdf() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    const evaluationId = generatePdfFromHistory.dataset.id;
    const evaluation = evaluations.find(e => e.id === evaluationId);
    if (!evaluation) return;

    // Título
    doc.setFontSize(18);
    doc.setTextColor(46, 125, 50); // Color hospital-red
    doc.text('MedPredict Pro - Ficha Clínica', 105, 20, { align: 'center' });
    
    // Logo (simulado)
    doc.setFontSize(12);
    doc.setTextColor(0, 0, 0);
    doc.text(`ID: ${evaluation.id}`, 15, 15);
    
    // Datos del paciente
    doc.setFontSize(14);
    doc.text('Datos del Paciente', 15, 35);
    doc.setFontSize(12);
    doc.text(`Nombre: ${evaluation.lastName}, ${evaluation.firstName}`, 15, 45);
    doc.text(`Edad: ${evaluation.age} años`, 15, 55);
    doc.text(`Fecha de evaluación: ${formatDate(evaluation.date)}`, 15, 65);
    
    // Datos clínicos
    doc.setFontSize(14);
    doc.text('Datos Clínicos', 15, 85);
    doc.setFontSize(12);
    doc.text(`Presión arterial: ${evaluation.bloodPressure} mmHg`, 15, 95);
    doc.text(`Frecuencia cardíaca: ${evaluation.heartRate} lpm`, 15, 105);
    doc.text(`Saturación de oxígeno: ${evaluation.oxygenLevel}%`, 15, 115);
    doc.text(`Condiciones crónicas: ${getChronicConditionsText(evaluation.chronicConditions)}`, 15, 125);
    
    // Resultados
    doc.setFontSize(14);
    doc.text('Resultados', 15, 145);
    doc.setFontSize(12);
    doc.text(`Nivel de riesgo: ${getRiskText(evaluation.riskLevel)}`, 15, 155);
    doc.text(`Nivel de severidad: ${evaluation.severityLevel}`, 15, 165);
    doc.text(`Probabilidad de mortalidad: ${evaluation.mortalityRisk}%`, 15, 175);
    
    // Factores de riesgo
    doc.setFontSize(14);
    doc.text('Factores de Riesgo', 15, 195);
    doc.setFontSize(12);
    const splitText = doc.splitTextToSize(evaluation.riskFactors || 'No se identificaron factores de riesgo adicionales.', 180);
    doc.text(splitText, 15, 205);
    
    // Pie de página
    doc.setFontSize(10);
    doc.setTextColor(100, 100, 100);
    doc.text('© 2025 MedPredict Pro - Sistema de Evaluación de Riesgo', 105, 285, { align: 'center' });
    
    doc.save(`Ficha_${evaluation.lastName}_${evaluation.firstName}.pdf`);
}

// Funciones auxiliares
function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString('es-ES', options);
}

function getRiskBadgeClass(riskLevel) {
    switch (riskLevel) {
        case 'low': return 'bg-success';
        case 'medium': return 'bg-warning text-dark';
        case 'high': return 'bg-danger';
        default: return 'bg-secondary';
    }
}

function getRiskText(riskLevel) {
    switch (riskLevel) {
        case 'low': return 'Bajo';
        case 'medium': return 'Medio';
        case 'high': return 'Alto';
        default: return 'Desconocido';
    }
}

function getRiskAlertClass(riskLevel) {
    switch (riskLevel) {
        case 'low': return 'alert-success';
        case 'medium': return 'alert-warning';
        case 'high': return 'alert-danger';
        default: return 'alert-secondary';
    }
}

function getChronicConditionsText(conditions) {
    const num = parseInt(conditions);
    if (num === 0) return 'Ninguna';
    if (num === 1) return '1 condición';
    if (num === 5) return '5 o más condiciones';
    return `${num} condiciones`;
}