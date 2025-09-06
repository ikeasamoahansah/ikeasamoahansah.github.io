module.exports = {
    content: [
      './src/**/*.{js,ts,jsx,tsx}',
      './templates/**/*.html',
    ],
    theme: {
      extend: {},
    },
    plugins: [
      require('@tailwindcss/typography'),
    ],
  };