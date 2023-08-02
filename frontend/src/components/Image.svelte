<script>
    import { dataStore } from '../store.js';

    let imageUrl = '';

    const unsubscribe = dataStore.subscribe(value => {
        if (value) {
            imageUrl = `http://localhost:8000/${value}`;
        }
    });
</script>

<style type="text/postcss">
    @tailwind base;
    @tailwind components;
    @tailwind utilities;
</style>

<div class="h-screen w-screen flex justify-center items-center">
    {#if imageUrl !== ''}
        <img src={imageUrl} alt="upload or generation from the prompt."/>
    {:else}
        <span class="dark:text-gray-100 outline-dashed dark:outline-gray-100 h-52 w-72 flex items-center text-center px-10 font-black">
            drag and drop image to upload or generate with prompt
        </span>
    {/if}
</div>
