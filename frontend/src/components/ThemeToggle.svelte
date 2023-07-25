<script>
  let darkMode = false;

  function handleSwitchDarkMode() {
    darkMode = !darkMode;

    darkMode
      ? document.body.classList.add("dark")
      : document.body.classList.remove("dark");
  }
</script>

<div>
  <input checked={darkMode} on:click={handleSwitchDarkMode} type="checkbox" id="theme-toggle" />
  <label for="theme-toggle" />
</div>

<style type="postcss">
  @tailwind base;
  @tailwind components;
  @tailwind utilities;

  #theme-toggle {
    @apply invisible;
  }

  #theme-toggle + label {
    @apply inline-block cursor-pointer h-12 w-12 absolute top-2 right-4 rounded-full duration-300 content-[''];
  }

  #theme-toggle:not(:checked) + label {
    @apply bg-orange-400;
  }

  #theme-toggle:checked + label {
    @apply bg-transparent;
    box-shadow: inset -18px -16px 1px 1px #ddd;
  }
</style>
