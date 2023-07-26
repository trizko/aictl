<script>
    let darkMode = true;
    darkMode
            ? document.body.classList.add("dark")
            : document.body.classList.remove("dark");

    function handleSwitchDarkMode() {
        darkMode = !darkMode;

        darkMode
            ? document.body.classList.add("dark")
            : document.body.classList.remove("dark");
    }
</script>

<div>
    <input
        checked={darkMode}
        on:click={handleSwitchDarkMode}
        type="checkbox"
        id="theme-toggle"
    />
    <label for="theme-toggle">
        <svg
            id="sun-icon"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="-12 -11 124 124"
            x="0px"
            y="0px"
            fill="#FFFFFF"
        >
            <g data-name="light, sun">
                <circle cx="50" cy="50" r="18" />
                <path d="M27,50a2,2,0,0,0-2-2H9a2,2,0,0,0,0,4H25A2,2,0,0,0,27,50Z"/>
                <path d="M91,48H75a2,2,0,0,0,0,4H91a2,2,0,0,0,0-4Z" />
                <path d="M50,27a2,2,0,0,0,2-2V9a2,2,0,0,0-4,0V25A2,2,0,0,0,50,27Z"/>
                <path d="M50,73a2,2,0,0,0-2,2V91a2,2,0,0,0,4,0V75A2,2,0,0,0,50,73Z"/>
                <path d="M67.68,34.32a2,2,0,0,0,1.41-.58L80.41,22.42a2,2,0,1,0-2.83-2.83L66.26,30.91a2,2,0,0,0,1.42,3.41Z"/>
                <path d="M30.91,66.26,19.59,77.58A2,2,0,0,0,21,81a2,2,0,0,0,1.41-.58L33.74,69.09a2,2,0,0,0-2.83-2.83Z"/>
                <path d="M30.91,33.74a2,2,0,0,0,1.41.58,2,2,0,0,0,1.42-3.41L22.42,19.59a2,2,0,0,0-2.83,2.83Z"/>
                <path d="M69.09,66.26a2,2,0,0,0-2.83,2.83L77.58,80.41A2,2,0,0,0,79,81a2,2,0,0,0,1.42-3.41Z"/>
            </g>
        </svg>
        <svg
            id="moon-icon"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="-12 -11 124 124"
            x="0px"
            y="0px"
            fill="#FFFFFF"
        >
            <g data-name="dark, moon">
                <path d="M85.46,65.33a2,2,0,0,0-1.89-.85,33.84,33.84,0,0,1-34.1-49.27,2,2,0,0,0-2-2.91A37.85,37.85,0,1,0,85.58,67.4,2,2,0,0,0,85.46,65.33Z" />
            </g>
        </svg>
    </label>
</div>

<style type="postcss">
    @tailwind base;
    @tailwind components;
    @tailwind utilities;

    #theme-toggle {
        @apply hidden;
    }

    #theme-toggle + label {
        @apply bg-gray-500 cursor-pointer h-10 w-10 absolute top-2 right-2 rounded content-[''];
    }

    #theme-toggle:not(:checked) + label > #sun-icon {
        @apply hidden;
    }

    #theme-toggle:checked + label > #moon-icon {
        @apply hidden;
    }
</style>
