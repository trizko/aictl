<script>
    // You can add your JavaScript code here
</script>

<style>
    :global(body) {
        margin: 0;
        padding: 0;
        background: #000;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(50px, 1fr));
        grid-auto-rows: 50px;
        grid-gap: 5px;
    }

    .grid-item {
        background: rgba(255, 255, 255, 0.1);
    }
</style>

<main>
    {#each Array(100).fill() as _}
        <div class="grid-item"></div>
    {/each}
</main>
