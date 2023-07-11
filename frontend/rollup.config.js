import svelte from 'rollup-plugin-svelte';
import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';
import livereload from 'rollup-plugin-livereload';
import css from 'rollup-plugin-css-only';
import terser from '@rollup/plugin-terser';
import { spawn } from 'child_process';
import sveltePreprocess from 'svelte-preprocess'

const production = !process.env.ROLLUP_WATCH;

function serve() {
	let server;

	function toExit() {
		if (server) server.kill(0);
	}

    return {
        writeBundle() {
          if (server) return;
          // Spawn a child server process
          server = spawn(
            'npm',
            ['run', 'start', '--', '--dev'],
            {
              stdio: ['ignore', 'inherit', 'inherit'],
              shell: true,
            }
          );
    
          // Kill server on process termination or exit
          process.on('SIGTERM', toExit);
          process.on('exit', toExit);
        },
      };
}

export default {
	input: 'src/main.js',
	output: {
		sourcemap: true,
		format: 'iife',
		name: 'app',
		file: 'public/build/bundle.js'
	},
	plugins: [
		svelte({
			preprocess: sveltePreprocess({ sourceMap: !production, postcss: true }),
			compilerOptions: {
				dev: !production
			}
		}),
		css({ output: 'bundle.css' }),
		resolve({
			browser: true,
			dedupe: ['svelte']
		}),
		commonjs(),
		!production && serve(),
		!production && livereload('public'),
        production && terser(),
	],
	watch: {
		clearScreen: false
	}
};
