/* Unified Theme Toggle System for MarkRemoverAI
 * Handles light/dark mode switching with localStorage persistence
 * Works across all pages of the website
 */

(function() {
    'use strict';

    const root = document.documentElement;
    const themeToggleBtn = document.getElementById('themeToggle');

    /**
     * Apply theme to the page
     * @param {string} theme - 'light' or 'dark'
     */
    function applyTheme(theme) {
        if (theme === 'dark') {
            root.setAttribute('data-theme', 'dark');
            if (themeToggleBtn) {
                themeToggleBtn.textContent = 'Light';
                themeToggleBtn.setAttribute('aria-label', 'Switch to light mode');
            }
        } else {
            root.removeAttribute('data-theme');
            if (themeToggleBtn) {
                themeToggleBtn.textContent = 'Dark';
                themeToggleBtn.setAttribute('aria-label', 'Switch to dark mode');
            }
        }
    }

    /**
     * Get current theme from localStorage or system preference
     * @returns {string} 'light' or 'dark'
     */
    function getCurrentTheme() {
        try {
            // Check localStorage first
            const stored = localStorage.getItem('theme');
            if (stored === 'light' || stored === 'dark') {
                return stored;
            }
        } catch (e) {
            console.warn('localStorage not available:', e);
        }

        // Fall back to system preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        return prefersDark ? 'dark' : 'light';
    }

    /**
     * Toggle between light and dark themes
     */
    function toggleTheme() {
        const currentTheme = root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        applyTheme(newTheme);

        // Save preference
        try {
            localStorage.setItem('theme', newTheme);
        } catch (e) {
            console.warn('Could not save theme preference:', e);
        }
    }

    // Initialize theme on page load
    const initialTheme = getCurrentTheme();
    applyTheme(initialTheme);

    // Attach toggle event listener
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', toggleTheme);
    }

    // Listen for system theme changes
    try {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
            // Only auto-switch if user hasn't manually set a preference
            if (!localStorage.getItem('theme')) {
                applyTheme(e.matches ? 'dark' : 'light');
            }
        });
    } catch (e) {
        console.warn('Could not listen to system theme changes:', e);
    }

    // Expose theme API for debugging
    window.ThemeAPI = {
        get: getCurrentTheme,
        set: applyTheme,
        toggle: toggleTheme
    };
})();
