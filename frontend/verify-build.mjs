import { access } from 'node:fs/promises';

const requiredFiles = [
  new URL('./dist/index.html', import.meta.url),
  new URL('./dist/logo.png', import.meta.url),
];

await Promise.all(requiredFiles.map(file => access(file)));
console.log('PlanetRead frontend build is ready in dist/.');
