<script>
    async function generateImage() {
        const response = await fetch("http://localhost:8000/generate");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        data = await response.json();

        return data
    }
</script>

<style type="text/postcss">
    @tailwind base;
    @tailwind components;
    @tailwind utilities;
</style>

<div class="bg-gray-500 px-2 py-2">
    <input class="py-1 px-1" type="text" />
</div>
<button on:click={generateImage} class="text-gray-900 bg-orange-400 rounded-r text-xl font-black px-5 py-1 hover:bg-orange-300">generate</button>
