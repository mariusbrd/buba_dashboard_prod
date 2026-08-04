// Custom tooltip transform functions for Dash components
window.dccFunctions = window.dccFunctions || {};

// Convert decimal year (e.g., 2024.75) to quarter format (e.g., "2024 Q4")
window.dccFunctions.decimalYearToQuarter = function (value) {
    const year = Math.floor(value);
    const fraction = value - year;

    let quarter;
    if (fraction < 0.125) {
        quarter = 1;
    } else if (fraction < 0.375) {
        quarter = 1;
    } else if (fraction < 0.625) {
        quarter = 2;
    } else if (fraction < 0.875) {
        quarter = 3;
    } else {
        quarter = 4;
    }

    return year + ' Q' + quarter;
};
