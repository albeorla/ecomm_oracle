import {chromium} from "playwright-extra";
import StealthPlugin from "puppeteer-extra-plugin-stealth";
// import UAPlugin from "puppeteer-extra-plugin-anonymize-ua";

chromium.use(StealthPlugin());
// chromium.use(UAPlugin());

async function main() {
    const credentials = {
        email: "albertjorlando@gmail.com",
        password: "uad*rfy0jcz0EBM0med"
    };
    const url = "https://members.junglescout.com/#/opportunity-finder";

    const browser = await chromium.launch({
        headless: false,
        devtools: true,
        slowMo: 1000,
    });

    const page = await browser.newPage();
    await page.goto(url, {waitUntil: "networkidle"});

}

main().then(() => console.log("done"));
