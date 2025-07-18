// /** @type {import('postcss').ProcessOptions} */
// const config = {
//   plugins: {
//     tailwindcss: {},
//     autoprefixer: {},
//   },
// };

// export default config;
// import tailwindcss from "tailwindcss"
// import autoprefixer from "autoprefixer"

// export default {
//   plugins: [tailwindcss, autoprefixer]
// }

// export default {
//   plugins: {
//     tailwindcss: {},
//     autoprefixer: {},
//   },
// }


// postcss.config.mjs
import tailwindcss from "tailwindcss"
import autoprefixer from "autoprefixer"

export default {
  plugins: [tailwindcss, autoprefixer],
}
