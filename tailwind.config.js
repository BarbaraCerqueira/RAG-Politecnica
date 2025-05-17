/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // You can customize these colors to match PoliGPT's branding
        blue: {
          100: '#E6F0FF',
          200: '#CCE0FF',
          300: '#99C0FF',
          400: '#66A0FF',
          500: '#3380FF',
          600: '#0055CC',
          700: '#004099',
          800: '#002B66',
          900: '#001533',
        },
        gold: {
          100: '#FFF7E6',
          200: '#FFEFC0',
          300: '#FFE399',
          400: '#FFD670',
          500: '#FFCC4D',
          600: '#E6AC00',
          700: '#B38600',
          800: '#805F00',
          900: '#4D3900',
        },
      },
      animation: {
        fadeIn: 'fadeIn 0.3s ease-in-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: 0, transform: 'translateY(10px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [
    require('tailwind-scrollbar')({ nocompatible: true }),
  ],
};