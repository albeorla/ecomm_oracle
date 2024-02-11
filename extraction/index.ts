import {chromium} from "playwright-extra";

import StealthPlugin from "puppeteer-extra-plugin-stealth";

chromium.use(StealthPlugin());

async function main() {
    const url = "https://nowsecure.nl";
    const browser = await chromium.launch({headless: false});
    const page = await browser.newPage();
    await page.goto(url, {waitUntil: "networkidle"});
}

main().then(() => console.log("done"));
